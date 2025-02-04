from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from copy import deepcopy

import torch

import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward

import logging
logging.basicConfig(level=logging.INFO)

torch.set_float32_matmul_precision('high')

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):
    # Deal with feature things before anything
    opt.use_att = utils.if_use_att(opt.caption_model)


    # Set random seed
    if opt.seed != -1:
        seed = opt.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # This is makes things deterministic
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    loader = DataLoader(opt)

    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, opt.INFOS_FILE_NAME), 'rb') as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, opt.HISTORIES_FILE_NAME)):
            with open(os.path.join(opt.start_from, opt.HISTORIES_FILE_NAME), 'rb') as f:
                histories = cPickle.load(f)

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    model = models.setup(opt).cuda()

    # https://pytorch.org/docs/2.6/torch.compiler_troubleshooting.html#where-to-apply-torch-compile
    # model = torch.compile(model, 
    #                       dynamic=True)
    dp_model = torch.nn.DataParallel(model)

    epoch_done = True
    # Assure in training mode
    dp_model.train()

    if opt.label_smoothing > 0:
        crit = utils.LabelSmoothing(smoothing=opt.label_smoothing)
    else:
        crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()

    if opt.noamopt:
        assert opt.caption_model in ['transformer', 
                                     'simple_transformer', 
                                     'simple_transformer_fc', 
                                     'simple_transformer_fc_multi_res',
                                     'simple_transformer_fc_multi_enc',
                                     'simple_transformer_moe',
                                     'simple_transformer_multi_enc_moe',
                                     ], 'noampt can only work with transformer models'
        optimizer = utils.get_std_opt(model, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
        optimizer._step = iteration
    elif opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    else:
        optimizer = utils.build_optimizer(model.parameters(), opt)
    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from, opt.OPTIMISER_FILE_NAME)):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from,  opt.OPTIMISER_FILE_NAME)))

    sc_flag = False
    best_val_score = 0.0
    while True:
        if epoch_done:
            if not opt.noamopt and not opt.reduce_on_plateau:
                # Assign the learning rate
                if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                    frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                    decay_factor = opt.learning_rate_decay_rate  ** frac
                    opt.current_lr = opt.learning_rate * decay_factor
                else:
                    opt.current_lr = opt.learning_rate
                utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            epoch_done = False

        # Load data from train split (0)
        data = loader.get_batch('train')

        torch.cuda.synchronize()
        start = time.time()

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp
        labels = labels.to(torch.int64)

        slots = {}
        # NOTE: Handle sending slots to GPU
        # TODO: Magic strings should be pulled out into a single place as constants
        if opt.caption_model not in ["simple_transformer","simple_transformer_fc"]:
            slots = data["slots"]

            for i, res in enumerate(slots['resolutions']):
                slots['resolutions'][i] = torch.from_numpy(res).float().cuda()

            for i, mask in enumerate(slots['masks']):
                slots['masks'][i] = torch.from_numpy(mask).float().cuda()

        optimizer.zero_grad()
        reward = 0

        if not sc_flag:
            loss = crit(dp_model(fc_feats, att_feats, deepcopy(slots), labels, att_masks), labels[:,1:], masks[:,1:])
        else:
            gen_result, sample_logprobs = dp_model(fc_feats, att_feats, slots, att_masks, opt={'sample_max':0}, mode='sample')
            reward = get_self_critical_reward(dp_model, fc_feats, att_feats, slots, att_masks, data, gen_result, opt)
            loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda())

        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        train_loss = loss.item()
        torch.cuda.synchronize()
        end = time.time()
        if not sc_flag:
            logging.info("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}, CIDEr = {:.3f}" \
                .format(iteration, epoch, train_loss, end - start, best_val_score))
        else:
            logging.info("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}, CIDEr = {:.3f}" \
                .format(iteration, epoch, np.mean(reward[:,0]), end - start, best_val_score))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            epoch_done = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            if opt.noamopt:
                opt.current_lr = optimizer.rate()
            elif opt.reduce_on_plateau:
                opt.current_lr = optimizer.current_lr

            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:,0])
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(dp_model, crit, loader, eval_kwargs)

            if opt.reduce_on_plateau:
                if 'CIDEr' in lang_stats:
                    optimizer.scheduler_step(-lang_stats['CIDEr'])
                else:
                    optimizer.scheduler_step(val_loss)

            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True: # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True

        optimizer.zero_grad()
        reward = 0


        if not sc_flag:
            loss = crit(dp_model(fc_feats, att_feats, deepcopy(slots), labels, att_masks), labels[:,1:], masks[:,1:])
        else:
            gen_result, sample_logprobs = dp_model(fc_feats, att_feats, slots, att_masks, opt={'sample_max':0}, mode='sample')
            reward = get_self_critical_reward(dp_model, fc_feats, att_feats, slots, att_masks, data, gen_result, opt)
            loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda())

        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        train_loss = loss.item()
        torch.cuda.synchronize()
        end = time.time()
        if not sc_flag:
            logging.info("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}, CIDEr = {:.3f}" \
                .format(iteration, epoch, train_loss, end - start, best_val_score))
        else:
            logging.info("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}, CIDEr = {:.3f}" \
                .format(iteration, epoch, np.mean(reward[:,0]), end - start, best_val_score))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            epoch_done = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            if opt.noamopt:
                opt.current_lr = optimizer.rate()
            elif opt.reduce_on_plateau:
                opt.current_lr = optimizer.current_lr

            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:,0])
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(dp_model, crit, loader, eval_kwargs)

            if opt.reduce_on_plateau:
                if 'CIDEr' in lang_stats:
                    optimizer.scheduler_step(-lang_stats['CIDEr'])
                else:
                    optimizer.scheduler_step(val_loss)

            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True: # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True

                if not os.path.isdir(opt.checkpoint_path):
                    os.makedirs(opt.checkpoint_path)
                checkpoint_path = os.path.join(opt.checkpoint_path, opt.MODEL_FILE_NAME)
                torch.save(model.state_dict(), checkpoint_path)
                logging.info("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, opt.OPTIMISER_FILE_NAME)
                torch.save(optimizer.state_dict(), optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history


                with open(os.path.join(opt.checkpoint_path, opt.INFOS_FILE_NAME), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, opt.HISTORIES_FILE_NAME), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, opt.BEST_MODEL_FILE_NAME)
                    torch.save(model.state_dict(), checkpoint_path)
                    logging.info("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, opt.BEST_INFOS_FILE_NAME), 'wb') as f:
                        cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

opt = opts.parse_opt()
train(opt)

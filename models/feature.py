import torch


def get_bert_features(bert_encoder, dataset, device, win_w, win_len=2, max_pool=False, concat_cls=False):

    bert_encoder = bert_encoder.to(device)

    features_list = {}

    for split in ['Train', 'Dev', 'Test']:

        features = []
        for _, sent in enumerate(dataset[split]):

            # batch_size, seq_len = sent['input_ids'].shape
            input_ids = torch.tensor(sent['input_ids'], device=device).reshape(1, -1)  #含[CLS][SEP], token_len*1
            encoder_output = bert_encoder(input_ids)  #encoder_output:token_len * 768
            #word level, 0:cls
            encoder_hidden_state = torch.mm(torch.tensor(sent['input_mask'], device=device).T, encoder_output['last_hidden_state'].squeeze())  # (n_sent, n_word, 768)

            masks = []

            word_masks = torch.eye(encoder_hidden_state.shape[0], device=device)[1:-1]  #去掉[CLS][SEP], 0索引对应第0token(不含[CLS])

            for _, event in enumerate(sent['output_event']):
                mention_mask = torch.zeros((1, encoder_hidden_state.shape[0]-2), device=device)  #不含[CLS][SEP]的索引
                mention_mask[:, event['tokens_number']] = 1  #'token_number':起始0 不考虑cls, encoder_hidden_state 需要+1
                if win_len>0 and win_w>0:
                    mention_mask = expand_win(event['tokens_number'], mention_mask, win_w, win_len)
                
                mask_sum = mention_mask.sum(dim=1)
                mask_ave = (1 / mask_sum).repeat((1, mention_mask.shape[1]))
                mention_mask = mention_mask * mask_ave

                this_mask = torch.mm(mention_mask, word_masks)  #mention mask(考虑了cls)

                masks.append(this_mask)
            for _,entity in enumerate(sent['output_entity']):

                mention_mask = torch.zeros((1, encoder_hidden_state.shape[0]-2), device=device)  #不含[CLS][SEP]的索引
                mention_mask[:, entity['tokens_number']] = 1
                if win_len>0 and win_w>0:
                    mention_mask =  expand_win(entity['tokens_number'], mention_mask, win_w, win_len)

                mask_sum = mention_mask.sum(dim=1)
                mask_ave = (1 / mask_sum).repeat((1, mention_mask.shape[1]))
                mention_mask = mention_mask * mask_ave

                this_mask = torch.mm(mention_mask, word_masks)

                masks.append(this_mask)
                
            masks = torch.cat(masks, dim=0).cuda()
            # print("##1", encoder_hidden_state.shape)
            # encoder_hidden_state = encoder_hidden_state.squeeze()  #?
            # print("##2", encoder_hidden_state.shape)
            
            # if self.cls: 拼接encoder_hidden_state[0]
            this_feature = torch.mm(masks, encoder_hidden_state)
            if concat_cls:
                this_feature = torch.concat((this_feature, encoder_hidden_state[0:1, :].repeat((this_feature.shape[0], 1))), dim=1)
            features.append(this_feature)

        features_list[split] = torch.cat(features).detach()
    
    return features_list


def expand_win(numbers, mask, win_w, win_len):
    max_count, count, l, max_l, max_r = 0, 0, numbers[0], numbers[0], numbers[-1]
    nums = [numbers[0]-1]+numbers+[numbers[-1]+2]
    for i in range(1, len(nums)-1):
        
        if nums[i] - nums[i-1] > 1:  #不连续
            if count>max_count:
                max_count = count
                max_l = l
                max_r = nums[i-1]
            l = nums[i]
            count=1
        else:
            count += 1
    expand = mask.clone()
    expand[:, max(max_l-win_len, 0):max_l] = win_w
    expand[:, max_r+1:min(max_r+1+win_len, mask.shape[-1])] = win_w
    return torch.where(mask>0, mask, expand)

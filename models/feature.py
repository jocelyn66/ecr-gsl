import torch


def get_bert_features(bert_encoder, dataset, device):

    bert_encoder = bert_encoder.to(device)

    features_list = {}

    for split in ['Train', 'Dev', 'Test']:

        features = []
        for _, sent in enumerate(dataset[split]):

            # batch_size, seq_len = sent['input_ids'].shape

            input_ids = torch.tensor(sent['input_ids'], device=device).reshape(1, -1)
            encoder_output = bert_encoder(input_ids)
            encoder_hidden_state = encoder_output['last_hidden_state']  # (n_sent, n_tokens, feat_dim)

            masks = []
            # token_masks = torch.tensor(sent['input_mask'])
            token_masks = torch.eye(input_ids.shape[1], device=device)

            for _, event in enumerate(sent['output_event']):
                this_mask = token_masks[event['tokens_number']]
                this_mask = torch.mean(this_mask, dim=0, keepdim=True)
                masks.append(this_mask)
            for _,entity in enumerate(sent['output_entity']):
                this_mask = torch.mean(token_masks[entity['tokens_number']], dim=0, keepdim=True)
                masks.append(this_mask)
                
            masks = torch.cat(masks, dim=0).cuda()
            encoder_hidden_state = encoder_hidden_state.squeeze()
            
            features.append(torch.mm(masks, encoder_hidden_state))

        features_list[split] = torch.cat(features).detach()
    
    return features_list

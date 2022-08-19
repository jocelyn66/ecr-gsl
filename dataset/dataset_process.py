import torch

#Process Datasets
def preprocess_function(examples, tokenizer, args, schema_list, plm=None):
    doc_schema = schema_list[0]
    event_schema = schema_list[1]
    entity_schema = schema_list[2]

    length = len(examples['tokens'])
    for k in examples.data.keys():
        assert len(examples[k]) == length

    input_ids = []
    input_mask = []
    input_doc = []
    input_sent = []
    output_event = []
    output_entity = []
    for i in range(length):
        #添加token和分词后token对应的序号
        text = examples['text'][i]
        tokens = examples['tokens'][i]

        tokens_embed = [tokenizer.cls_token_id]
        tokened_idx = [0]
        idx = 1
        for t in tokens:
            embed = tokenizer(t)['input_ids'][1:-1] #[1:-1]是为了去除开头的CLS和结尾的SEP
            tokens_embed.extend(embed)
            tokened_idx.append(idx)
            idx += len(embed)
        tokens_embed.append(tokenizer.sep_token_id) #SEP的id
        tokened_idx.append(idx) #SEP的起始
        tokened_idx.append(idx+1) #SEP的结束

        token_mask = torch.zeros((len(tokens_embed), len(tokens)+2))
        for j in range(len(tokened_idx)-1):
            token_mask[tokened_idx[j]:tokened_idx[j+1], j] = 1
        #将每列的1替换为均值权重，以保证mask的效果等价为average_pooling
        mask_sum = token_mask.sum(dim=0)
        mask_ave = (1 / mask_sum).repeat((token_mask.size()[0], 1))
        token_mask = token_mask * mask_ave
        
        input_ids.append(tokens_embed)
        input_mask.append(token_mask)

        #添加文档编号和句子编号        
        input_doc.append(doc_schema[examples['doc_id'][i]])
        input_sent.append(int(examples['sent_id'][i]))
        
        #添加事件coref
        event_coref = examples['event_coref'][i]
        temp_eve = []
        for eve in event_coref:
            eve_new = {'coref_chain':event_schema[eve['coref_chain']], 'tokens_number':eve['tokens_number']}
            temp_eve.append(eve_new)
        output_event.append(temp_eve)
        
        #添加实体coref
        entity_coref = examples['entity_coref'][i]
        temp_ent = []
        for ent in entity_coref:
            ent_new = {'coref_chain':entity_schema[ent['coref_chain']], 'tokens_number':ent['tokens_number']}
            temp_ent.append(ent_new)
        output_entity.append(temp_ent)
        
    return {'input_ids':input_ids, 'input_mask':input_mask, 'input_doc':input_doc, 'input_sent':input_sent, 
            'output_event':output_event, 'output_entity':output_entity}  
            # 'input_ids':token_len * 1
            # 'input_mask':input_mask: token_len * word_len, 
            # 'input_doc':input_doc:一个id, 
            # 'input_sent':input_sent:一个id, 
            # 'output_event':output_event: 索引list的list, 对应的索引是word级别的
            # 'output_entity':output_entity} :一个list, 对应的索引是word级别的
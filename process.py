import numpy as np
import torch
import wordninja
import random
import re
from torch.utils.data import Dataset, DataLoader


class sequence_data_sampler_bert_prompt(object):

    def __init__(self, data_sampler, config, seed=None):
        self.data_sampler = data_sampler
        self.batch = 0
        self.len = data_sampler.num_clusters
        if data_sampler.seed != None:
            random.seed(data_sampler.seed)

        self.shuffle_index = list(range(self.len))
        random.shuffle(self.shuffle_index)
        self.shuffle_index = np.argsort(self.shuffle_index)
        print(self.shuffle_index)
        self.id2rel, self.rel2id = data_sampler.id2rel, data_sampler.relation_names
        self.seen_relations = []
        self.history_test_data = {}
        self.history_test_data_old = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch == self.len:
            self.batch = 0
            raise StopIteration()
        
        
        index = self.shuffle_index[self.batch]
        self.batch += 1
        training_data = self.data_sampler.splited_training_data[index]
        valid_data = self.data_sampler.splited_valid_data[index]
        test_data = self.data_sampler.splited_test_data[index]

        current_relations = []
        cur_training_data_2 = {}
        cur_valid_data_2 = {}
        cur_test_data_2 = {}
        indexs = []

        for data in training_data:
            if data[0] not in self.seen_relations:
                self.seen_relations.append(self.id2rel[data[0]])
            if data[0] not in current_relations:
                current_relations.append(self.id2rel[data[0]])
                indexs.append(data[0])
        
        cur_training_data = self.remove_unseen_relation(training_data, self.seen_relations)
        cur_valid_data = self.remove_unseen_relation(valid_data, self.seen_relations)
        self.history_test_data_old.append(test_data)



        current_relations = list(set(current_relations))
        self.seen_relations = list(set(self.seen_relations))
        indexs = list(set(indexs))
        
        
        cur_test_data = []
        for j in range(self.batch):
            cur_test_data.append(self.remove_unseen_relation(self.history_test_data_old[j], self.seen_relations))

        
        # Cho nay moi la cho setup data
        for index in indexs:

            cur_training_data_2[self.id2rel[index]] = []
            cur_valid_data_2[self.id2rel[index]] = []
            cur_test_data_2[self.id2rel[index]] = []
            self.history_test_data[self.id2rel[index]] = []
            
            for itm in cur_training_data:
                if itm['relation'] == index:
                    cur_training_data_2[self.id2rel[index]].append(itm)
                    
            for itm in cur_valid_data:
                if itm['relation'] == index:
                    cur_valid_data_2[self.id2rel[index]].append(itm)
                    
            for itm in cur_valid_data:
                if itm['relation'] == index:
                    cur_test_data_2[self.id2rel[index]].append(itm)
                    
            for itm in cur_valid_data:
                if itm['relation'] == index:
                    self.history_test_data[self.id2rel[index]].append(itm)

        
        return cur_training_data_2, cur_valid_data_2, cur_test_data_2, current_relations, self.history_test_data, self.seen_relations

    def __len__(self):
        return self.len

    def remove_unseen_relation(self, dataset, seen_relations):
        cleaned_data = []
        for data in dataset:
            neg_cands = [cand for cand in data[1] if cand in seen_relations and cand != data[0]]
            if len(neg_cands) > 0:
                cleaned_data.append([data[0], neg_cands, data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15]])
            else:
                # if self.data_sampler.config['task_name'] == 'FewRel' or self.data_sampler.config['task_name'] == "TacRed":
                cleaned_data.append([data[0], data[1][-2:], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15]])

        new_cleaned_data = []
        for item in cleaned_data:
            new_cleaned_data.append({
                'relation': item[0],
                'neg_labels': item[1],
                'tokens': item[2],
            })
        
        return new_cleaned_data




class data_sampler_bert_prompt_deal_first_task(object):

    def __init__(self, config=None, tokenizer=None, template=None, max_length=128):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template

        self.training_data = self._gen_data(config.training_file)
        self.valid_data = self._gen_data(config.valid_file)
        self.test_data = self._gen_data(config.test_file)

        self.relation_names, self.id2rel = self._read_relations(config.relation_file)
        print('Relation_file:', config.relation_file)
        self.id2rel_pattern = {}
        for i in self.id2rel:
            tokens, fakefirstent, fakefirstindex, fakesecondent, fakesecondindex, fakeheadid, faketailid, rawtext, length, fakelabel, mask = self._transfrom_sentence(self.id2rel[i])
            self.id2rel_pattern[i] = (i, [i], tokens, fakefirstent, fakefirstindex, fakesecondent, fakesecondindex, fakeheadid, faketailid, rawtext, length, fakelabel, mask, 0, fakefirstindex, fakesecondindex) ####need new format
        self.num_clusters = 8
        self.cluster_labels = {}
        self.rel_features = {}
        rel_index = np.load(config.rel_index)

        rel_cluster_label = np.load(config.rel_cluster_label)
        print("Rel_cluster_label: ", config.rel_cluster_label)
        rel_feature = np.load(config.rel_feature)
        for index, i in enumerate(rel_index):
            self.cluster_labels[i] = rel_cluster_label[index]
            self.rel_features[i] = rel_feature[index]
            
        self.splited_training_data = self._split_data(self.training_data, self.cluster_labels, self.num_clusters)

        self.splited_training_data = self._deal_first_task(self.splited_training_data)


        self.splited_valid_data = self._split_data(self.valid_data, self.cluster_labels, self.num_clusters)
        self.splited_test_data = self._split_data(self.test_data, self.cluster_labels, self.num_clusters)
        self.seed = None
    
    def _deal_first_task(self, splited_training_data):
        splited_training_data_list = []
        for training_data in splited_training_data:
            if len(training_data) > self.config.first_task_k_way * self.config.k_shot:
                task_data = []
                class_dic = {}
                for data in training_data:
                    if data[0] in class_dic.keys():
                        if class_dic[data[0]] < self.config.k_shot:
                            class_dic[data[0]] += 1
                            task_data.append(data)
                    else:
                        class_dic[data[0]] = 1
                        task_data.append(data)
                splited_training_data_list.append(task_data)
            else:
                splited_training_data_list.append(training_data)
        return splited_training_data_list

    def _split_data(self, data_set, cluster_labels, num_clusters):
        splited_data = [[] for i in range(num_clusters)]
        for data in data_set:
            splited_data[cluster_labels[data[0]]].append(data)
        return splited_data

    def _gen_data(self, file):
        data = self._read_samples(file)
        data = self._transform_questions(data)
        print(data[0])
        return data

    def _read_samples(self, file):
        sample_data = []
        with open(file) as file_in:
            for line in file_in:
                items = line.strip().split('\t')
                if (len(items[0]) > 0):
                    relation_ix = int(items[0])
                    if items[1] != 'noNegativeAnswer':
                        candidate_ixs = [int(ix) for ix in items[1].split()]
                        question = self._remove_return_sym(items[2])
                        firstent = items[3]
                        firstentindex = [int(ix) for ix in items[4].split()]
                        secondent = items[5]
                        secondentindex = [int(ix) for ix in items[6].split()]
                        headid = items[7]
                        tailid = items[8]
                        sample_data.append(
                            [relation_ix, candidate_ixs, question, firstent, firstentindex, secondent, secondentindex, headid, tailid])
        return sample_data

    def handletoken(self, raw_text, h_pos_li, t_pos_li, tokenizer, err):
        h_pattern = re.compile("\* h \*")
        t_pattern = re.compile("\^ t \^")
        tokens = []
        h_mention = []
        t_mention = []
        raw_text_list = raw_text.split(" ")
        for i, token in enumerate(raw_text_list):
            token = token.lower()
            if i >= h_pos_li[0] and i <= h_pos_li[-1]:
                if i == h_pos_li[0]:
                    tokens += ['*', 'h', '*']
                h_mention.append(token)
                continue
            if i >= t_pos_li[0] and i <= t_pos_li[-1]:
                if i == t_pos_li[0]:
                    tokens += ['^', 't', '^']
                t_mention.append(token)
                continue
            tokens.append(token)
        text = " ".join(tokens)
        h_mention = " ".join(h_mention)
        t_mention = " ".join(t_mention)

        temp = self.template
        mask = self.tokenizer.mask_token
        temp = temp.replace('mask', mask)
        temp = temp.replace('e1', h_mention)
        temp = temp.replace('e2', t_mention)
        text = temp + text

        tokenized_text = tokenizer.tokenize(text)
        tokenized_head = tokenizer.tokenize(h_mention)
        tokenized_tail = tokenizer.tokenize(t_mention)

        p_text = " ".join(tokenized_text)
        p_head = " ".join(tokenized_head)
        p_tail = " ".join(tokenized_tail)

        ifoldmethod = True

        if ifoldmethod:
            p_text = h_pattern.sub("[unused0] " + p_head + " [unused1]", p_text)
            p_text = t_pattern.sub("[unused2] " + p_tail + " [unused3]", p_text)
        else:
            p_text = h_pattern.sub(p_head, p_text)
            p_text = t_pattern.sub(p_tail, p_text)

        f_text = ("[CLS] " + p_text + " [SEP]").split()

        mask_pos = f_text.index(mask)

        try:
            if ifoldmethod:
                h_begin_new = f_text.index("[unused0]") + 1
                h_end_new = f_text.index("[unused1]") - 1
            else:
                h_begin_new = -1
                h_end_new = -1
                for aa in range(len(f_text)):
                    if f_text[aa] == tokenized_head[0]:
                        h_begin_new = aa
                        if f_text[aa + len(tokenized_head) - 1] == tokenized_head[-1]:
                            h_end_new = aa + len(tokenized_head) - 1
                            break
                assert (h_end_new - h_begin_new) + 1 == len(tokenized_head)
        except:
            err += 1
            h_begin_new = 0
            h_end_new = 0
        try:
            if ifoldmethod:
                t_begin_new = f_text.index("[unused2]") + 1
                t_end_new = f_text.index("[unused3]") - 1
            else:
                t_begin_new = -1
                t_end_new = -1
                for aa in range(len(f_text)):
                    if f_text[aa] == tokenized_tail[0]:
                        t_begin_new = aa
                        if f_text[aa + len(tokenized_tail) - 1] == tokenized_tail[-1]:
                            t_end_new = aa + len(tokenized_tail) - 1
                            break
                assert (t_end_new - t_begin_new) + 1 == len(tokenized_tail)
        except:
            err += 1
            t_begin_new = 0
            t_end_new = 0

        g_text = " ".join(f_text)
        l_text = g_text.split()
        tokenized_input = tokenizer.convert_tokens_to_ids(l_text)

        return tokenized_input, h_begin_new, h_end_new, t_begin_new, t_end_new, mask_pos

    def _transform_questions(self, data):
        err = 0
        for sample in data:
            originaltext = sample[2]
            h_pos_li = sample[4]
            t_pos_li = sample[6]
            tokenized_input, h_begin_new, h_end_new, t_begin_new, t_end_new, mask_pos = self.handletoken(originaltext,h_pos_li,t_pos_li,self.tokenizer,err)
            thislength = min(len(tokenized_input), self.max_length)
            if (len(tokenized_input) > self.max_length):
                tokenized_input = tokenized_input[0:self.max_length]
            newtokeninput = []
            for i in range(0,thislength):
                newtokeninput.append(tokenized_input[i])
            for i in range(thislength,self.max_length):
                newtokeninput.append(0)
            sample[2] = newtokeninput

            firstent = sample[3]

            newsample4 = []
            for aa in range(h_begin_new,h_end_new+1):
                newsample4.append(aa)
            sample[4] = newsample4
            fentsize = len(sample[4])
            firstenttoken = self.tokenizer.tokenize(firstent)
            oldfirstsize = len(firstenttoken)
            firstenttoken = self.tokenizer.convert_tokens_to_ids(firstenttoken)
            newfirstsize = len(firstenttoken)

            assert oldfirstsize == newfirstsize
            sample[3] = firstenttoken

            secondent = sample[5]
            newsample6 = []
            for aa in range(t_begin_new,t_end_new+1):
                newsample6.append(aa)
            sample[6] = newsample6
            sentsize = len(sample[6])
            secondenttoken =  self.tokenizer.tokenize(secondent)
            oldsecondsize = len(secondenttoken)
            secondenttoken = self.tokenizer.convert_tokens_to_ids(secondenttoken)
            newsecondsize = len(secondenttoken)

            assert oldsecondsize == newsecondsize
            sample[5] = secondenttoken

            sample.append(originaltext)

            sample.append(thislength)

            typelabel = 1
            sample.append(typelabel)

            mask = []
            for i in range(0, thislength):
                mask.append(1)
            for i in range(thislength, self.max_length):
                mask.append(0)
            sample.append(mask)
            sample.append(mask_pos)

            sample.append(h_pos_li)
            sample.append(t_pos_li)
        print("final err: ",err)
        return data

    def _transfrom_sentence(self, data):
        rawtext = data
        touse = "[CLS] " + data + " [SEP]"
        tokens = self.tokenizer.tokenize(touse)

        tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        length = min(len(tokens), self.max_length)
        if (len(tokens) > self.max_length):
            tokens = tokens[:self.max_length]
        newtokens = []
        for i in range(0, length):
            newtokens.append(tokens[i])
        for i in range(length, self.max_length):
            newtokens.append(0)
        fakefirstent = [554,555]
        fakefirstindex = [0,1]
        fakesecondent = [665,666]
        fakesecondindex = [3,4]
        fakeheadid = "fheadid"
        faketailid = "ftailid"
        fakelabel = 0 
        mask = []
        for i in range(0, length):
            mask.append(1)
        for i in range(length, self.max_length):
            mask.append(0)

        return newtokens, fakefirstent, fakefirstindex, fakesecondent, fakesecondindex, fakeheadid, faketailid, rawtext, length, fakelabel, mask

    def __iter__(self):
        return sequence_data_sampler_bert_prompt(self, self.seed)

    def set_seed(self, seed):
        self.seed = seed

    def _remove_return_sym(self, str):
        return str.split('\n')[0]

    def _read_relations(self, file):
        relation_list = [self._split_relation_into_words(self._remove_return_sym('fill fill fill'))]
        id2rel = {0: 'fill fill fill'}
        with open(file) as file_in:
            for line in file_in:
                relation_list.append(self._split_relation_into_words(self._remove_return_sym(line)))
                id2rel[len(id2rel)] = self._remove_return_sym(line)
        return relation_list, id2rel

    def _split_relation_into_words(self, relation):
        word_list = []
        for word_seq in relation.split("/")[-3:]:
            for word in word_seq.split("_"):
                word_list += wordninja.split(word)
        return " ".join(word_list)




class data_set_bert_prompt(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, data):
        #print(data[0])
        labels = torch.tensor([item[0] for item in data])
        neg_labels = [torch.tensor(item[1]) for item in data]
        sentences = torch.stack([torch.tensor(item[2]) for item in data]) #
        #print(sentences.shape)
        #print(sentences)
        firstent = [torch.tensor(item[3]) for item in data]
        firstentindex = [torch.tensor(item[4]) for item in data]
        secondent = [torch.tensor(item[5]) for item in data]
        secondentindex = [torch.tensor(item[6]) for item in data]
        headid = [item[7] for item in data]
        tailid = [item[8] for item in data]
        rawtext = [item[9] for item in data]
        lenghts = [torch.tensor(item[10]) for item in data]
        typelabels =  torch.tensor([item[11] for item in data])
        masks = torch.stack([torch.tensor(item[12]) for item in data]) #
        mask_pos = torch.tensor([item[13] for item in data]) #
        #print(masks.shape)
        #print(masks)
        return (
            labels,
            neg_labels,
            sentences,
            # firstent,
            # firstentindex,
            # secondent,
            # secondentindex,
            # headid,
            # tailid,
            # rawtext,
            # lenghts,
            # typelabels,
            # masks,
            # mask_pos
        )


def get_data_loader_bert_prompt(config, data, shuffle = True, drop_last = False, batch_size = None):
    dataset = data_set_bert_prompt(data)
    if batch_size == None:
        batch_size = min(config.batch_size_per_step, len(data))
    else:
        batch_size = min(batch_size, len(data))

    data_loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        pin_memory = True,
        num_workers = config.num_workers,
        collate_fn = dataset.collate_fn,
        drop_last = drop_last)
    return data_loader
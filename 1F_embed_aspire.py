# use sciemb conda env: conda activate sciemb
import sys
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import nltk
import numpy as np
import torch
from datetime import datetime
import json
import glob


class DataEncoder:

    def __init__(self, model='specter_hf'):
        # Init the tokenizer and encoding the model
        if model == 'specter_hf':
            self.load_specter_hf()
        elif model == 'scibert':
            self.load_scibert()
        elif model == 'sbert':
            self.load_sbert()
        elif model == 'specter_scratch':
            self.load_specter_scratch()
        elif model == 'aspire':
            self.load_aspire()

        # self.model.to('cuda:1')
        self.model.eval()
        return

    def update_encoding_model(self, model_name):
        if model_name == 'specter_hf':
            self.load_specter()
        elif model_name == 'scibert':
            self.load_scibert()
        elif model_name == 'sbert':
            self.load_sbert()
        elif model_name == 'specter_scratch':
            self.load_specter_scratch()
        elif model_name == 'aspire':
            self.load_aspire()
        else:
            print('Invalid model name! Please try again with a valid  model name.')

    def load_specter_hf(self):
        self.model_name = 'specter_hf'
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        self.model = AutoModel.from_pretrained('allenai/specter')
        return

    def load_specter_scratch(self):
        self.model_name = "specter_scratch"
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        self.model = AutoModel.from_pretrained('./trained_models/specter_scratch/', local_files_only=True)
        return

    def load_aspire(self):
        self.model_name = "aspire"
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/aspire-biencoder-compsci-spec')
        self.model = AutoModel.from_pretrained('allenai/aspire-biencoder-compsci-spec')
        return

    def load_scibert(self):
        self.model_name = 'scibert'
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')
        self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_cased')
        return

    def load_sbert(self):
        self.model_name = 'sbert'
        self.model = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens')
        return
    
    def encode_batch_wise_using_scibert(self, document_dict, to_save_loc):
        document_emb_dict = {}

        for _, d in enumerate(document_dict):
            if _ % 2000 == 0:
                print("Encoded: {} @ {}".format(_, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            # document_list is actually a dictionatry. org_pid_seq is not provided
            text_to_encode = (document_dict[d].get('title') or '') + self.tokenizer.sep_token + (document_dict[d].get('abstract') or '')
            document_id = d

            inputs = self.tokenizer(text_to_encode, padding=True, truncation=True, return_tensors="pt", max_length=512)
            result = self.model(**inputs)
            embedding = result.last_hidden_state[:, 0, :].detach().numpy().tolist()
            document_emb_dict[document_id] = embedding
            
            if _%4000 == 0:
                print("Dumping {} lines to file.".format(_))
                with open(to_save_loc, 'a') as outfile:
                    for _, entry in enumerate(document_emb_dict):
                        try:
                            outfile.write(json.dumps({"paper_id": entry, "title": (document_dict[entry].get('title') or ''), "embedding": document_emb_dict[entry]}) + '\n')
                        except:
                            print(_, entry)
                            continue
                document_emb_dict = {}

        with open(to_save_loc, 'a') as outfile:
            for _, entry in enumerate(document_emb_dict):
                try:
                    outfile.write(json.dumps({"paper_id": entry, "embedding": document_emb_dict[entry]}) + '\n')
                except:
                    print(_, entry)
                    continue

        return document_emb_dict
    
    def encode_batch_wise_using_sbert(self, document_dict, to_save_loc):
        document_emb_dict = {}

        for _, d in enumerate(document_dict):
            if _ % 2000 == 0:
                print("Encoded: {} @ {}".format(_, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            # document_list is actually a dictionatry. org_pid_seq is not provided
            text_to_encode = []
            if document_dict[d].get('title'):
                text_to_encode = [document_dict[d].get('title')]
            if document_dict[d].get('abstract'):
                abs_sents = nltk.sent_tokenize(document_dict[d].get('abstract'))
                text_to_encode += abs_sents
            document_id = d

            sbert_embeddings = self.model.encode(text_to_encode)
            embeddings_sum = np.sum(sbert_embeddings, axis=0)
            l2_normed_embedding = embeddings_sum / np.sqrt(np.sum(embeddings_sum**2))
            document_emb_dict[document_id] = l2_normed_embedding.tolist()
            
            if _%4000 == 0:
                print("Dumping 4k lines to file.")
                with open(to_save_loc, 'a') as outfile:
                    for _, entry in enumerate(document_emb_dict):
                        try:
                            outfile.write(json.dumps({"paper_id": entry, "title": (document_dict[entry].get('title') or ''), "embedding": document_emb_dict[entry]}) + '\n')
                        except:
                            print(_, entry)
                            continue
                document_emb_dict = {}

        with open(to_save_loc, 'a') as outfile:
            for _, entry in enumerate(document_emb_dict):
                try:
                    outfile.write(json.dumps({"paper_id": entry, "title": (document_dict[entry].get('title') or ''), "embedding": document_emb_dict[entry]}) + '\n')
                except:
                    print(_, entry)
                    continue
        return document_emb_dict

    def encode_batch_wise_using_oagbert(self, document_dict, to_save_loc):
        document_emb_dict = {}

        for _, d in enumerate(document_dict):
            if _ % 2000 == 0:
                print("Encoded: {} @ {}".format(_, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            # document_list is actually a dictionatry. org_pid_seq is not provided
            paper_title = (document_dict[d].get('title') or '')
            paper_abstract = (document_dict[d].get('abstract') or '')
            document_id = d

            #check if title+abs is empty
            title_abs = paper_title + paper_abstract
            if not title_abs.strip():
                continue

            input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, \
            masked_positions, num_spans = self.model.build_inputs(
                title=paper_title, abstract=paper_abstract, venue=[], authors=[], concepts=[], affiliations=[])
            sequence_output, pooled_output = self.model.bert.forward(
                input_ids=torch.LongTensor(input_ids).unsqueeze(0),
                token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0),
                attention_mask=torch.LongTensor(input_masks).unsqueeze(0),
                output_all_encoded_layers=False,
                checkpoint_activations=False,
                position_ids=torch.LongTensor(position_ids).unsqueeze(0),
                position_ids_second=torch.LongTensor(position_ids_second).unsqueeze(0)
            )

            embedding = pooled_output.detach().numpy().tolist()
            document_emb_dict[document_id] = embedding

            if _%4000 == 0:
                print("Dumping 4k lines to file.")
                with open(to_save_loc, 'a') as outfile:
                    for _, entry in enumerate(document_emb_dict):
                        try:
                            outfile.write(json.dumps({"paper_id": entry, "title": (document_dict[entry].get('title') or ''), "embedding": document_emb_dict[entry]}) + '\n')
                        except:
                            print(_, entry)
                            continue
                document_emb_dict = {}

        with open(to_save_loc, 'a') as outfile:
            for _, entry in enumerate(document_emb_dict):
                try:
                    outfile.write(json.dumps({"paper_id": entry, "title": (document_dict[entry].get('title') or ''), "embedding": document_emb_dict[entry]}) + '\n')
                except:
                    print(_, entry)
                    continue

        return document_emb_dict

    def encode_batch_wise_using_specter_scratch(self, document_dict, to_save_loc, batch_size=20):
        # List to contain small batch of douments and the corresponding doc ids
        doc_batch_list = []
        batch_ids = []

        document_emb_dict = {}

        for _, d in enumerate(document_dict):
            if _ % 2000 == 0:
                print("Encoded: {} @ {}".format(_, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

            doc_batch_list.append((document_dict[d].get('title') or '') + self.tokenizer.sep_token + (document_dict[d].get('abstract') or ''))
            batch_ids.append(d)

            if _%batch_size == 0:
                inputs = self.tokenizer(doc_batch_list, padding=True, truncation=True, return_tensors="pt", max_length=512)
                # inputs = inputs.to('cuda:1')
                result = self.model(**inputs)
                embeddings = result.last_hidden_state[:, 0, :]

                for ii, k in enumerate(batch_ids):
                    document_emb_dict[k] = embeddings[ii].detach().cpu().numpy().reshape(768,).tolist()

                doc_batch_list = []
                batch_ids = []

            if _%4000 == 0:
                print("Dumping {} lines to file.".format(_))
                with open(to_save_loc, 'a') as outfile:
                    for _, entry in enumerate(document_emb_dict):
                        try:
                            outfile.write(json.dumps({"paper_id": entry, "title": (document_dict[entry].get('title') or ''), "embedding": document_emb_dict[entry]}) + '\n')
                        except:
                            print(_, entry)
                            continue
                document_emb_dict = {}

        if batch_ids:
            inputs = self.tokenizer(doc_batch_list, padding=True, truncation=True, return_tensors="pt", max_length=512)
            # inputs = inputs.to('cuda:1')
            result = self.model(**inputs)
            embeddings = result.last_hidden_state[:, 0, :]

            for _, k in enumerate(batch_ids):
                document_emb_dict[k] = embeddings[_].detach().cpu().numpy().reshape(768,).tolist()

        # To freeup memory in case of reuse in jupyter
        doc_batch_list = []
        batch_ids = []

        with open(to_save_loc, 'a') as outfile:
            for _, entry in enumerate(document_emb_dict):
                try:
                    outfile.write(json.dumps({"paper_id": entry, "title": (document_dict[entry].get('title') or ''), "embedding": document_emb_dict[entry]}) + '\n')
                except:
                    print(_, entry)
                    continue
        return document_emb_dict

    def encode_batch_wise_using_aspire(self, document_dict, to_save_loc, batch_size=20):
        # List to contain small batch of douments and the corresponding doc ids
        doc_batch_list = []
        batch_ids = []

        document_emb_dict = {}

        for _, d in enumerate(document_dict):
            if _ % 2000 == 0:
                print("Encoded: {} @ {}".format(_, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

            doc_batch_list.append((document_dict[d].get('title') or '') + self.tokenizer.sep_token + (document_dict[d].get('abstract') or ''))
            batch_ids.append(d)

            if _%batch_size == 0:
                inputs = self.tokenizer(doc_batch_list, padding=True, truncation=True, return_tensors="pt", max_length=512)
                # inputs = inputs.to('cuda:1')
                result = self.model(**inputs)
                embeddings = result.last_hidden_state[:, 0, :]

                for ii, k in enumerate(batch_ids):
                    document_emb_dict[k] = embeddings[ii].detach().cpu().numpy().reshape(768,).tolist()

                doc_batch_list = []
                batch_ids = []

            if _%4000 == 0:
                print("Dumping {} lines to file.".format(_))
                with open(to_save_loc, 'a') as outfile:
                    for _, entry in enumerate(document_emb_dict):
                        try:
                            outfile.write(json.dumps({"paper_id": entry, "title": (document_dict[entry].get('title') or ''), "embedding": document_emb_dict[entry]}) + '\n')
                        except:
                            print(_, entry)
                            continue
                document_emb_dict = {}

        if batch_ids:
            inputs = self.tokenizer(doc_batch_list, padding=True, truncation=True, return_tensors="pt", max_length=512)
            # inputs = inputs.to('cuda:1')
            result = self.model(**inputs)
            embeddings = result.last_hidden_state[:, 0, :]

            for _, k in enumerate(batch_ids):
                document_emb_dict[k] = embeddings[_].detach().cpu().numpy().reshape(768,).tolist()

        # To freeup memory in case of reuse in jupyter
        doc_batch_list = []
        batch_ids = []

        with open(to_save_loc, 'a') as outfile:
            for _, entry in enumerate(document_emb_dict):
                try:
                    outfile.write(json.dumps({"paper_id": entry, "title": (document_dict[entry].get('title') or ''), "embedding": document_emb_dict[entry]}) + '\n')
                except:
                    print(_, entry)
                    continue
        return document_emb_dict
    
    def encode_batch_wise_using_specter(self, document_dict, to_save_loc, batch_size=20):
        # List to contain small batch of douments and the corresponding doc ids
        doc_batch_list = []
        batch_ids = []

        document_emb_dict = {}

        for _, d in enumerate(document_dict):
            if _ % 2000 == 0:
                print("Encoded: {} @ {}".format(_, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

            doc_batch_list.append((document_dict[d].get('title') or '') + self.tokenizer.sep_token + (document_dict[d].get('abstract') or ''))
            batch_ids.append(d)

            if _%batch_size == 0:
                inputs = self.tokenizer(doc_batch_list, padding=True, truncation=True, return_tensors="pt", max_length=512)
                # inputs = inputs.to('cuda:1')
                result = self.model(**inputs)
                embeddings = result.last_hidden_state[:, 0, :]

                for ii, k in enumerate(batch_ids):
                    document_emb_dict[k] = embeddings[ii].detach().cpu().numpy().tolist()

                doc_batch_list = []
                batch_ids = []

        if batch_ids:
            inputs = self.tokenizer(doc_batch_list, padding=True, truncation=True, return_tensors="pt", max_length=512)
            # inputs = inputs.to('cuda:1')
            result = self.model(**inputs)
            embeddings = result.last_hidden_state[:, 0, :]

            for _, k in enumerate(batch_ids):
                document_emb_dict[k] = embeddings[_].detach().cpu().numpy().tolist()

        # To freeup memory in case of reuse in jupyter
        doc_batch_list = []
        batch_ids = []

        with open(to_save_loc, 'w') as outfile:
            for _, entry in enumerate(document_emb_dict):
                try:
                    outfile.write(json.dumps({"paper_id": entry, "title": (document_dict[entry].get('title') or ''), "embedding": document_emb_dict[entry]}) + '\n')
                except:
                    print(_, entry)
                    continue
        return document_emb_dict

if __name__ == "__main__":

    file_idx = int(sys.argv[1])

    de = DataEncoder(model='aspire')

    all_task_files = glob.glob("./data/*.json")

    for f in [all_task_files[file_idx]]:
        print("Processing file: {}".format(f))
        with open(f, 'r') as fin:
            data_dict = json.load(fin)

            out_file_name = f.rsplit("/", 1)[-1]
            out_file_name = out_file_name.replace("paper_metadata_", "")
            out_file_name = out_file_name.replace(".json", "")
            
            sd = de.encode_batch_wise_using_aspire(data_dict, './data/trained_embs/aspire-embeddings/{}.jsonl'.format(out_file_name))
import math
import tensorflow_hub as hub
import tensorflow_text as text

class BERT:



    def load_BERT(self):
        preprocess_url = "data/model/bert_preprocessor"  # "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
        encoder_url = "data/model/bert"  # "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

        self.bert_preprocessor = hub.KerasLayer(preprocess_url)
        self.bert_model = hub.KerasLayer(encoder_url)


    def _calculate_BERT_embeddings(self, labels):
        bert_embedding = []
        batch_size = 100
        for batch in range(0, math.ceil(len(labels) / batch_size)):
            start, end = batch * 100, batch * 100 + batch_size
            labels_slice = labels[start:end]
            preprocessed_text = self.bert_preprocessor(labels_slice)
            bert_results = self.bert_model(preprocessed_text)
            label_vectors = bert_results['pooled_output']

            for idx in range(len(labels_slice)):
                bert_embedding.append(label_vectors[idx].numpy())

            print(f'\r calculating bert embeddings:{batch / (len(labels) // batch_size):.2f}', end='')

        print()

        return bert_embedding


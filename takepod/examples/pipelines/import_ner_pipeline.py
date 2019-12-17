import pickle
from takepod.examples.pipelines.ner_pipeline import CroatianNERPipeline


# keras model load
# pipeline = pickle.load(open('ner_pipeline.pkl', 'rb'))

pipeline = CroatianNERPipeline()
pipeline.load_all()



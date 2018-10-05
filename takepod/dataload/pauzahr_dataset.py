import os
import glob
import xml.etree.ElementTree as ET

from takepod.storage import dataset
from takepod.storage.example import Example
from takepod.storage.field import Field

PauzaHrFields = [Field('text', use_vocab=True),
                 Field('source', use_vocab=False), 
                 Field('rating', use_vocab=False)]

class PauzaHr(dataset.Dataset):
    urls = ["http://takelab.fer.hr/data/cropinion/CropinionDataset.zip"]
    
    def __init__(self, fields, path="downloaded_datasets/", train=True):
        self._data_dir = path
        

        datalocation = os.path.join(
            self._data_dir, "croopinion",
            "CropinionDataset", "reviews_original"
        )
        if train:
            datalocation = os.path.join(datalocation, "Train")
        # only train vs. test split
        elif not train:
            datalocation = os.path.join(datalocation, "Test")



        data = []
        labels = []
        for xmlfile in glob.glob(datalocation + "/*.xml"):
            root = ET.parse(xmlfile)
            review_text = root.find('Text').text
            source = root.find('Source').text
            rating = root.find('Rating').text
            
            data.append((review_text, source))
            labels.append(float(rating))

        super(PauzaHr, self).__init__()
"""Module for loading raw eurovoc dataset"""
import glob
import logging
import os
import xml.etree.ElementTree as ET
from collections import namedtuple
from enum import Enum

import dill

from podium.datasets.impl.eurovoc_dataset import EuroVocDataset
from podium.storage.resources.large_resource import (
    LargeResource,
    init_scp_large_resource_from_kwargs,
)


_LOGGER = logging.getLogger(__name__)
try:
    import xlrd
except ImportError:
    _LOGGER.debug(
        "Problem occured while trying to import xlrd. If the "
        "library is not installed visit http://www.python-excel.org/ "
        "for more details."
    )

Document = namedtuple("Document", "filename title text")


class LabelRank(Enum):
    """Levels of labels in EuroVoc."""

    THESAURUS = 3
    MICRO_THESAURUS = 2
    TERM = 1


class Label:
    """Label in EuroVoc dataset.

    Labels are assigned to documents. One document has multiple labels.
    Labels have a hierarchy in which one label can have one or more parents (broader
    terms). All labels apart from thesaurus rank labels have at least one parent.
    Apart from parents, labels can also have similar labels which describe related
    areas, but aren't connected by the label hierarchy.
    """

    def __init__(
        self,
        name,
        id,
        direct_parents,
        similar_terms,
        rank,
        thesaurus=None,
        micro_thesaurus=None,
        all_ancestors=None,
    ):
        """Defines a single label in the EuroVoc dataset.

        Parameters
        ----------
        name : str
            name of the label
        id : int
            numerical id of the label
        direct_parents : list(int)
            list of ids of direct parents
        similar_terms : list(int)
            list of ids of similar terms
        rank : LabelRank
            rank of the label
        thesaurus : int
            id of the thesaurus of the label (if the label represents a
            thesaurus, it has its own id listed in this field)
        micro_thesaurus : int
            id of the microthesaurus of the label (if the label represents
            a microthesaurus, it has its own id listed in this field)
        all_ancestors : set(int)
            set of ids of all ancestors of the label in the label hierarchy
        """
        self.name = name
        self.id = id
        self.direct_parents = direct_parents
        self.all_ancestors = all_ancestors
        self.similar_terms = similar_terms
        self.rank = rank
        self.micro_thesaurus = micro_thesaurus
        self.thesaurus = thesaurus


class EuroVocLoader:
    """Class for downloading and parsing the EuroVoc dataset.

    This class is used for downloading the EuroVoc dataset (if it's not already
    downloaded) and parsing the files in the dataset. If it's not already present
    LargeResource.BASE_RESOURCE_DIR, the dataset is automatically downloaded when an
    instance of EuroVocLoader is created. The downloaded resources can be parsed using
    the load_dataset method.
    """

    URL = "/proj/sci/uisusd/data/eurovoc_data/eurovoc.zip"
    EUROVOC_LABELS_FILENAME = "EUROVOC.xml"
    CROVOC_LABELS_FILENAME = "CROVOC.xml"
    MAPPING_FILENAME = "mapping.xlsx"
    DATASET_DIR = "Data"
    DOCUMENT_PATHS = "*.xml"
    SCP_HOST = "djurdja.takelab.fer.hr"
    ARCHIVE_TYPE = "zip"
    NAME = "EuroVocDataset"

    def __init__(self, **kwargs):
        """Constructor of the EuroVocLoader class.

        The constructor will check if the dataset is already been downloaded in the
        LargeResource.BASE_RESOURCE_DIR. If the dataset is not present, it will atempt to
        download it.

        **kwargs:
            SCPLargeResource.SCP_USER_KEY:
                Username on the host machine from which the dataset is downloaded. Not
                required if the username on the local machine matches the username on the
                host.
            SCPLargeResource.SCP_PRIVATE_KEY:
                Path to the ssh private key eligible to access the host. Not required on
                Unix if the private key is stored in the default location.
            SCPLargeResource.SCP_PASS_KEY:
                Password for the ssh private key (optional). Can be omitted
                if the private key is not encrypted.
        """
        init_scp_large_resource_from_kwargs(
            resource=EuroVocLoader.NAME,
            uri=EuroVocLoader.URL,
            user_dict=kwargs,
            archive=EuroVocLoader.ARCHIVE_TYPE,
            scp_host=EuroVocLoader.SCP_HOST,
        )

    def load_dataset(self):
        """Loads and parses all the necessary files from the dataset folder.

        Returns
        -------
        tuple:
            (EuroVoc label hierarchy, CroVoc label hierarchy, document mapping,
            documents)

            EuroVoc label hierarchy : dict(label_id : Label)
            CroVoc label hierarchy : dict(label_id : Label)
            document mapping : dict(document_id : list of label ids)
            documents : list(Document)
        """
        eurovoc_label_hierarchy_path = os.path.join(
            LargeResource.BASE_RESOURCE_DIR,
            EuroVocLoader.NAME,
            EuroVocLoader.EUROVOC_LABELS_FILENAME,
        )
        eurovoc_labels = EuroVocLoader._parse_label_hierarchy(
            eurovoc_label_hierarchy_path
        )

        crovoc_label_hierarchy_path = os.path.join(
            LargeResource.BASE_RESOURCE_DIR,
            EuroVocLoader.NAME,
            EuroVocLoader.CROVOC_LABELS_FILENAME,
        )
        crovoc_labels = EuroVocLoader._parse_label_hierarchy(crovoc_label_hierarchy_path)

        document_mapping_path = os.path.join(
            LargeResource.BASE_RESOURCE_DIR,
            EuroVocLoader.NAME,
            EuroVocLoader.MAPPING_FILENAME,
        )
        mapping = EuroVocLoader._parse_mappings(document_mapping_path)

        dataset_path = os.path.join(
            LargeResource.BASE_RESOURCE_DIR,
            EuroVocLoader.NAME,
            EuroVocLoader.DATASET_DIR,
            EuroVocLoader.DOCUMENT_PATHS,
        )
        documents = EuroVocLoader._parse_documents(dataset_path, mapping)

        return eurovoc_labels, crovoc_labels, mapping, documents

    @staticmethod
    def _parse_labels_by_name(label_hierarchy_path):
        """Does the first pass through the label file that maps label names to label ids.

        The label hierarchy is parsed from an xml file and returned as a dictionary where
        keys are label names and values are instances of Label class. This is done
        because in the original xml file labels are connected to other labels (e.g. as
        parents or similar terms) using their names. We wish to connect them using unique
        label ids instead.

        Parameters
        ----------
        label_hierarchy_path : path to xml file containing label hierarchy

        Returns
        -------
        tuple:
            (terms_by_name, microthesaurus_by_name, thesaurus_by_name, labels_by_id)

            terms_by_name : dict(term_name : term_id)
            microthesaurus_by_name : dict(microthesaurus_name : microthesaurus_id)
            thesaurus_by_name : dict(thesaurus_name : thesaurus_id)
            labels_by_id : dict(label_id : Label)
        """
        xml_document = label_hierarchy_path
        tree = ET.parse(xml_document)
        root = tree.getroot()

        # These dictionaries are used in the second pass for replacing string names with
        # label ids. Keys are string names and their values are ids.
        # Sometimes a term and a thesaurus or a microthesaurus may share the same name,
        # that's the reason for separate dictionaries for every label category.
        terms_by_name = {}
        microthesaurus_by_name = {}
        thesaurus_by_name = {}

        # This is the final label list that will eventually be used in the dataset.
        labels_by_id = {}

        for child in root:
            # If tag 'Podrucje' does not exist, it means this record is a thesaurus.
            if child.find("Podrucje") is None and child.find("Potpojmovnik") is None:
                rank = LabelRank.THESAURUS
                thesaurus = int(_get_text(child, "ID"))
                micro_thesaurus = None

            elif (
                child.find("Podrucje") is not None and child.find("Potpojmovnik") is None
            ):
                # If tag 'Podrucje' exists, but there is not 'Potpojmovnik' tag, it means
                # this record is a microthesaurus.
                rank = LabelRank.MICRO_THESAURUS
                thesaurus = _get_text(child, "Podrucje").split(";")[1]
                micro_thesaurus = int(_get_text(child, "ID"))

            elif (
                child.find("Podrucje") is not None
                and child.find("Potpojmovnik") is not None
            ):
                # If both 'Podrucje' and 'Potpojmovnik' tags exist, it means this record
                # is a term.
                rank = LabelRank.TERM
                thesaurus = _get_text(child, "Podrucje").split(";")[1]
                micro_thesaurus = _get_text(child, "Potpojmovnik")

            else:
                raise ValueError(
                    "Invalid label record. The record contains tag"
                    "<Potpojmovnik> but lacks the <Podrucje> tag."
                )

            name = child.find("Odrednica").text.lower().strip()
            label_id = int(_get_text(child, "ID"))

            parents = [
                broader_term.text.lower().strip()
                for broader_term in child.findall("SiriPojam")
            ]

            similar_terms = [
                similar_term.text.lower().strip()
                for similar_term in child.findall("SrodniPojam")
            ]

            # Here parents, similar terms, thesaurus and micro-thesaurus are all stored
            # using string names. In the second pass, these fields will be replaces by
            # matching ids.
            label = Label(
                name=name,
                id=label_id,
                direct_parents=parents,
                similar_terms=similar_terms,
                rank=rank,
                thesaurus=thesaurus,
                micro_thesaurus=micro_thesaurus,
            )
            labels_by_id[label_id] = label

            if rank == LabelRank.THESAURUS:
                thesaurus_by_name[name] = label

            elif rank == LabelRank.MICRO_THESAURUS:
                microthesaurus_by_name[name] = label

            elif rank == LabelRank.TERM:
                terms_by_name[name] = label

        return terms_by_name, microthesaurus_by_name, thesaurus_by_name, labels_by_id

    @staticmethod
    def _parse_label_hierarchy(label_hierarchy_path):
        """Parses the label hierarchy.

        The label hierarchy is parsed from an xml file and returned as a dictionary where
        keys are label ids and values are instances of Label class.

        Parameters
        ----------
        label_hierarchy_path : path to xml file containing label hierarchy

        Returns
        -------
        dict:
            Dictionary of (key, value) = (label_id, Label)
        """

        (
            terms_by_name,
            microthesaurus_by_name,
            thesaurus_by_name,
            labels_by_id,
        ) = EuroVocLoader._parse_labels_by_name(label_hierarchy_path)

        for label_id in labels_by_id:
            label = labels_by_id[label_id]

            # Names of the parents are replaced by their ids.
            # Parents can only be terms here, never thesaurus of microthesaurus.
            label.direct_parents = [
                terms_by_name[parent].id for parent in label.direct_parents
            ]

            # Names of the similar terms are replaced by their ids.
            # Similar terms can only be terms, never thesaurus of microthesaurus.
            label.similar_terms = [
                terms_by_name[similar_term].id for similar_term in label.similar_terms
            ]

            # If label is not thesaurus, replace its thesaurus name by thesaurus id
            if label.rank != LabelRank.THESAURUS:
                if label.thesaurus not in thesaurus_by_name:
                    # Error: thesaurus name does not exist (this shouldn't happen)
                    debug_msg = (
                        "Label {} has a non-existing thesaurus name"
                        "assigned: {}".format(label.id, label.thesaurus)
                    )
                    _LOGGER.debug(debug_msg)
                    label.thesaurus = None
                else:
                    label.thesaurus = thesaurus_by_name[label.thesaurus].id

            # If label is microthesaurus, then its thesaurus is listed as its parent
            if label.rank == LabelRank.MICRO_THESAURUS:
                if label.thesaurus:
                    label.direct_parents.append(label.thesaurus)

            # If label is term, replace its microthesaurus name by its id
            if label.rank == LabelRank.TERM:
                if label.micro_thesaurus not in microthesaurus_by_name:
                    # Error: microthesaurus name does not exist (this shouldn't happen)
                    debug_msg = (
                        "Label {} has a non-existing microthesaurus name"
                        "assigned: {}".format(label.id, label.micro_thesaurus)
                    )
                    _LOGGER.debug(debug_msg)
                    label.micro_thesaurus = None
                else:
                    label.micro_thesaurus = microthesaurus_by_name[
                        label.micro_thesaurus
                    ].id
                    # if term has no parent term then its microthesaurus is listed as
                    # its parent
                    if not label.direct_parents and label.micro_thesaurus:
                        label.direct_parents.append(label.micro_thesaurus)

        labels_by_id = EuroVocLoader._collect_all_ancestors(labels_by_id)
        return labels_by_id

    @staticmethod
    def _collect_all_ancestors(label_hierarchy):
        """Finds and stores the ancestors of all the labels in the label hierarchy.

        Parameters
        ----------
        label_hierarchy : dict(int, Label)
            Dictionary that maps label_id to Label.

        Returns
        -------
        dict:
            Dictionary of (key, value) = (label_id, Label)
        """

        new_label_hierarchy = {}

        for label_id in label_hierarchy:
            ancestors = EuroVocLoader._get_all_ancestors(label_id, label_hierarchy)
            label = label_hierarchy[label_id]
            new_label_hierarchy[label_id] = Label(
                name=label.name,
                id=label.id,
                rank=label.rank,
                direct_parents=label.direct_parents,
                similar_terms=label.similar_terms,
                thesaurus=label.thesaurus,
                micro_thesaurus=label.micro_thesaurus,
                all_ancestors=ancestors,
            )
        return new_label_hierarchy

    @staticmethod
    def _get_all_ancestors(label_id, label_hierarchy):
        """Finds and returns the ancestors of the label with the given label_id.

        Parameters
        ----------
        label_id : int

        label_hierarchy : dict(int, Label)
            Dictionary that maps label_id to Label.

        """
        direct_parents = label_hierarchy[label_id].direct_parents
        parents = set(direct_parents)
        # while the iterations of the loop find new, untraversed parents
        while direct_parents:
            new_parents = set()
            # for each parent, add all its parents to next iteration parents
            for label in direct_parents:
                label_direct_parents = label_hierarchy[label].direct_parents
                new_parents.update(
                    parent for parent in label_direct_parents if parent not in parents
                )
            parents.update(new_parents)
            direct_parents = new_parents
        return parents

    @staticmethod
    def _parse_mappings(mappings_path):
        """Parses the mappings of documents to labels from a xlsx file.

        Parameters
        ----------
        mappings_path : path to mappings in xlsx format

        Returns
        -------
        dict
            Dictionary of (key, value) = (document_id, list of label ids)
        """
        wb = xlrd.open_workbook(mappings_path)
        sheet_0 = wb.sheet_by_index(0)

        mappings = {}
        # row zero is table header, data starts from row 1
        row = 1
        while row < sheet_0.nrows:
            # cokumn 0 contains document id
            document_id = int(sheet_0.cell_value(row, 0))
            label_ids = []

            while True:
                # column 2 contains label id and label name split by semicolon character
                # (sometimes this field can be an empty string, in that case we simply
                # skip the row)
                label = sheet_0.cell_value(row, 2)
                if label:
                    label_id = label.split(";")[0]
                    label_id = label_id
                    label_ids.append(int(label_id))
                row += 1
                # If a row has an empty first column, then the row contains another label
                # for the previously seen document. When the first column is not empty,
                # we have read all the labels for the previous document and need to
                # switch to a new document.
                if row >= sheet_0.nrows or sheet_0.cell_value(row, 0):
                    break

            mappings[document_id] = label_ids
        return mappings

    @staticmethod
    def _parse_documents(path, document_mapping):
        """Parses xml documents from the given path.

        If the document_id is not present in the given document_mapping dictionary, the
        document is not parsed.

        Parameters
        ----------
        path : path that specifies all documents to be parsed
        document_mapping : dictionary of (key, value) = (document_id, list of label ids)

        Returns
        -------
        list
            List of parsed documents as Document type objects.
        """
        xml_documents = glob.glob(path)
        parsed_documents = []
        for doc in xml_documents:
            # if there is no mapping to labels for the document, the document can't be
            # used in the dataset and it's therefore not parsed
            # this happens often because certain categoried of documents are not maped to
            # labels
            filename = os.path.basename(doc)
            document_id = int(os.path.splitext(filename)[0].replace("NN", ""))
            if document_id not in document_mapping:
                debug_msg = "{} document id not found in document" "mappings.".format(
                    document_id
                )
                _LOGGER.debug(debug_msg)
                continue
            parsed_doc = EuroVocLoader._parse_document(doc)
            # parsed_doc is None if there's been an error on document text extraction
            if parsed_doc:
                parsed_documents.append(parsed_doc)
        debug_msg = "Succesfully parsed documents: {}/{}".format(
            len(parsed_documents), len(xml_documents)
        )
        _LOGGER.debug(debug_msg)
        return parsed_documents

    @staticmethod
    def _parse_document(doc):
        """Parses the given document from xml.

        Parameters
        ----------
        doc : path to document in xml format

        Returns
        -------
        Document
            Parsed document as instance of Document named tuple.
        """
        tree = ET.parse(doc)
        root = tree.getroot()
        root_children = list(root)
        head = root_children[0]
        body = root_children[1]

        filename = os.path.basename(doc)
        title_text = " ".join([t.text for t in head.iter() if t.text]).strip()
        # Proper header should begin with a digit have the following format:
        # "document_number date title"
        # This is true for 99% of the documents and excpetions are ignored
        if title_text and title_text[0].isdigit():
            title_text = " ".join(title_text.split(" ")[2:])
        else:
            debug_msg = "{} file contains invalid document title: {}".format(
                filename, title_text
            )
            _LOGGER.debug(debug_msg)
        title_text = title_text.lower().replace("\r", "").replace("\n", "")

        body_text = []
        for b in body.iter():
            if b.text:
                body_text.append(b.text)
            # everything after the </br> tag will end up in tail
            if b.tail and b.tail.strip():
                body_text.append(b.tail)
        body_text = "\n".join(body_text).lower()

        # If a document is stored as pdf in the database, the extraction process
        # generates an xml contaning the following string. Text for these documents is
        # not available and they are therefore ignored.
        if "postupak ekstrakcije teksta" in body_text:
            debug_msg = "{} XML file does not contain a valid text".format(filename)
            _LOGGER.debug(debug_msg)
            return None

        return Document(title=title_text, text=body_text, filename=filename)


def _get_text(child, filed_name):
    """Extracts and returns lowercase striped text from field with the given name.

    Parameters
    ----------
    child : Element
        Element contaning label record from XML file.

    field_name : str
        Name of the field to be extracted.

    Returns
    -------
    str : Lowercase striped contents of the field.
    """
    return child.find(filed_name).text.lower().strip()


def dill_dataset(output_path):
    """Downloads the EuroVoc dataset (if not already present) and stores the dataset in a
    dill file.

    Parameters
    ----------
    output_path : str
        Path to the file where the dataset instance will be stored.
    """
    loader = EuroVocLoader()
    eurovoc_labels, crovoc_labels, mapping, documents = loader.load_dataset()
    dataset = EuroVocDataset(
        documents=documents,
        mappings=mapping,
        eurovoc_labels=eurovoc_labels,
        crovoc_labels=crovoc_labels,
    )
    dataset.finalize_fields()

    with open(output_path, "wb") as output_file:
        dill.dump(dataset, output_file)

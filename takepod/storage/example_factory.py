"""Module containing the Example Factory method used to dynamically create example
classes used for storage in Dataset classes"""

import logging
import json
import csv

import xml.etree.ElementTree as ET
from takepod.storage.field import unpack_fields

_LOGGER = logging.getLogger(__name__)


class Example:
    """Method models one example with fields that hold
    (raw, tokenized) values and special fields with "_"
    at the end that can cache numericalized values"""

    def __init__(self, fieldnames):
        """Method initializes example with given list of
        fieldnames

        Parameters
        ----------
        fieldnames : list(str)
            list of field names
        """
        for fieldname in fieldnames:
            setattr(self, fieldname, None)

    def __repr__(self):
        attributes = [att for att in dir(self) if not att.startswith("__")]
        att_values = [f"{att}: {getattr(self, att, None)}" for att in attributes]
        att_string = "; ".join(att_values)
        return f"{self.__class__.__name__}[{att_string}] at {hex(id(self))}"


class ExampleFactory:
    """Class used to create Example instances. Every ExampleFactory dynamically creates
    its own example class definition optimised for the fields provided in __init__."""

    def __init__(self, fields):
        """
        Creates a new ExampleFactory instance.

        Parameters
        ----------
        fields : (dict | list)
                Can be either a dict mapping column names to Fields
                (or tuples of Fields), or a list of Fields (or tuples of Fields).
                A Field value of None means the corresponding column will
                be ignored.

        """
        if isinstance(fields, dict):
            self.fields = {input_value_name: fields_
                           for input_value_name, fields_
                           in fields.items()
                           if fields_ is not None}
        else:
            self.fields = fields

        self.fieldnames = [field.name for field in unpack_fields(fields)]

        # add cache data fields
        self.fieldnames += [f"{fieldname}_" for fieldname in self.fieldnames]

    def create_empty_example(self):
        """Method creates empty example with field names stored in example factory.

        Returns
        -------
        example : Example
            empty Example instance with initialized field names
        """
        return Example(self.fieldnames)

    def from_dict(self, data):
        example = self.create_empty_example()

        for key, fields in self.fields.items():
            val = data.get(key)
            set_example_attributes(example, fields, val)

        return example

    def from_list(self, data):
        example = self.create_empty_example()
        for value, field in filter(lambda el: el[1] is not None, zip(data, self.fields)):
            set_example_attributes(example, field, value)

        return example

    def from_xml_str(self, data):
        """Method creates and Example from xml string.

        Parameters
        ----------
        data : str
            XML formated string that contains the values of a single data
            instance, that are to be mapped to Fields.

        Returns
        -------
        Example
            An Example whose attributes are the given Fields created with the
            given column values. These Fields can be accessed by their names.

        Raises
        ------
        ValueError
            if the name is not contained in the xml string
        ParseError
            if there was a problem while parsing xml sting, invalid xml
        """
        example = self.create_empty_example()

        # we ignore columns with field mappings set to None
        items = filter(lambda el: el[1] is not None, self.fields.items())
        root = ET.fromstring(data)

        for name, field in items:
            node = root.find(name)

            if node is None:
                if root.tag == name:
                    node = root
                else:
                    error_msg = f"Specified name {name} was not found in the " \
                        "input data"
                    _LOGGER.error(error_msg)
                    raise ValueError(error_msg)

            val = node.text
            set_example_attributes(example, field, val)

        return example

    def from_json(self, data):
        """ Creates an Example from a JSON object and the
        corresponding fields.


        Parameters
        ----------
        data : str
            A string containing a single JSON object
            (key-value pairs surrounded by curly braces).

        Returns
        -------
        Example
            An Example whose attributes are the given Fields created with the
            given column values. These Fields can be accessed by their names.

        Raises
        ------
        ValueError
            if JSON doesn't contain key name
        """

        return self.from_dict(json.loads(data))

    def from_csv(self, data, field_to_index=None, delimiter=","):
        """ Creates an Example from a CSV line and a corresponding
            list or dict of Fields.

            Parameters
            ----------
            data : str
                A string containing a single row of values separated by the
                given delimiter.
            field_to_index : dict
                A dict that maps column names to their indices in the line of data.
                Only needed if fields is a dict, otherwise ignored.
            delimiter : str
                The delimiter that separates the values in the line of data.

            Returns
            -------
            Example
                An Example whose attributes are the given Fields created with the
                given column values. These Fields can be accessed by their names.
            """
        elements = next(csv.reader([data], delimiter=delimiter))

        if isinstance(self.fields, list):
            return self.from_list(elements)
        else:
            data_dict = {f: elements[idx] for f, idx in field_to_index.items()}
            return self.from_dict(data_dict)

    def from_fields_tree(self, data, subtrees=False):
        """ Creates an Example (or multiple Examples) from a string
        representing an nltk tree and a list of corresponding values.

        Parameters
        ----------
        data : str
            A string containing an nltk tree whose values are to be mapped
            to Fields.
        subtrees : bool
            A flag denoting whether an example will be created from every
            subtree in the tree (when set to True), or just from the whole
            tree (when set to False).

        Returns
        -------
        (Example | list)
            If subtrees was False, returns an Example whose attributes are
            the given Fields created with the given column values.
            These Fields can be accessed by their names.

            If subtrees was True, returns a list of such Examples for every
            subtree in the given tree.
        """
        from nltk.tree import Tree

        tree = Tree.fromstring(data)
        if subtrees:
            subtree_lists = map(tree_to_list, tree.subtrees())

            # an example is created for each subtree
            return [self.from_list(subtree_list) for subtree_list in
                    subtree_lists]
        else:
            return self.from_list(tree_to_list(tree))


def tree_to_list(tree):
    """Method joins tree leaves and label in one list.

    Parameters
    ----------
    tree : tree
        nltk tree instance

    Returns
    -------
    tree_list : list
        tree represented as list with its label
    """
    return [' '.join(tree.leaves()), tree.label()]


def set_example_attributes(example, field, val):
    """Method sets example attributes with given values.

    Parameters
    ----------
    example : Example
        example instance to which we are setting attributes
    field : (Field|tuple(Field))
        field instance or instances that we are mapping
    val : str
        field value
    """

    def set_example_attributes_for_single_field(example, field, val):
        for name, data in field.preprocess(val):
            setattr(example, name, data)

    if isinstance(field, tuple):
        for f in field:
            set_example_attributes_for_single_field(example, f, val)

    else:
        set_example_attributes_for_single_field(example, field, val)

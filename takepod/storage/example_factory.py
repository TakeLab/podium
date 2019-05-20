"""Module containing the Example Factory method used to dynamically create example classes
used for storage in Dataset classes"""

import logging
import json
import csv

import xml.etree.ElementTree as ET

from recordclass import structclass
from uuid import uuid4

_LOGGER = logging.getLogger(__name__)


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

        fieldnames = tuple(field.name for field in unpack_fields(fields))

        # create unique class identifier required for pickling
        uid = uuid4()
        example_class_name = "Example_class_{}".format(str(uid).replace("-", "_"))

        self.example_class = structclass(example_class_name,
                                         fieldnames,
                                         defaults=[None] * len(fieldnames)
                                         )

        # add class object to globals so pickling can find the class definition
        globals()[example_class_name] = self.example_class

    def from_dict(self, data):
        example = self.example_class()

        for key, fields in self.fields.items():
            val = data.get(key)
            _set_example_attributes(example, fields, val)

        return example

    def from_list(self, data):
        example = self.example_class()
        for value, field in filter(lambda el: el[1] is not None, zip(data, self.fields)):
            _set_example_attributes(example, field, value)

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
        example = self.example_class()

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
            _set_example_attributes(example, field, val)

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


def unpack_fields(fields):
    """Flattens the given fields object into a flat list of fields.

    Parameters
    ----------
    fields : (list | dict)
        List or dict that can contain nested tuples and None as values and
        column names as keys (dict).

    Returns
    -------
    list[Field]
        A flat list of Fields found in the given 'fields' object.
    """

    unpacked_fields = list()

    fields = fields.values() if isinstance(fields, dict) else fields

    # None values represent columns that should be ignored
    for field in filter(lambda f: f is not None, fields):
        if isinstance(field, tuple):
            unpacked_fields.extend(field)
        else:
            unpacked_fields.append(field)

    return unpacked_fields


def _set_example_attributes(example, field, val):
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
    if isinstance(field, tuple):
        for f in field:
            setattr(example, f.name, f.preprocess(val))
    else:
        setattr(example, field.name, field.preprocess(val))

"""
Module containing the Example Factory method used to dynamically create example
classes used for storage in Dataset classes.
"""

import csv
import json
import xml.etree.ElementTree as ET
from enum import Enum
from typing import Union

from podium.utils.general_utils import repr_type_and_attrs


class ExampleFormat(Enum):
    LIST = "list"
    DICT = "dict"
    CSV = "csv"
    NLTK = "nltk"
    XML = "xml"
    JSON = "json"


FACTORY_METHOD_DICT = {
    "list": lambda data, factory: factory.from_list(data),
    "dict": lambda data, factory: factory.from_dict(data),
    "csv": lambda data, factory: factory.from_csv(data),
    "nltk": lambda data, factory: factory.from_fields_tree(data),
    "xml": lambda data, factory: factory.from_xml_str(data),
    "json": lambda data, factory: factory.from_json(data),
}


class Example(dict):
    """
    Base class for data instances in Podium.

    Each key corresponds to one Field and holds (raw, tokenized) values produced by
    that Field.

    To access the attributes, you can use either the dictionary syntax (obj[key]) or
    the standard attribute access (obj.key) if the key is a valid Python identifier.
    """

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self):
        attrs = {k: v for k, v in self.items() if not k.endswith("_")}
        return repr_type_and_attrs(self, attrs, with_newlines=True)

    @staticmethod
    def with_fields(fields):
        """
        Create an Example instance from the given fields. This function should
        be used in conjuction with the function that creates an Example from the
        specific format. See podium.storage.ExampleFactory for the concrete list
        of functions that create Examples from diffent formats.

        Notes
        -----
        This is a convenience function. Use podium.storage.ExampleFactory
        if performance is important.
        """
        return ExampleFactory(fields)


class ExampleFactory:
    """
    Class used to create Example instances.

    Every ExampleFactory dynamically creates its own example class definition
    optimised for the fields provided in __init__.
    """

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
            self.fields = {
                input_value_name: fields_
                for input_value_name, fields_ in fields.items()
                if fields_ is not None
            }
        else:
            self.fields = fields

    def from_dict(self, data):
        """
        Method creates example from data in dictionary format.

        Parameters
        ----------
        data : dict(str, object)
            dictionary that maps field name to field value

        Returns
        -------
        example : Example
            example instance with given data saved to fields
        """
        example = Example()

        for key, fields in self.fields.items():
            val = data.get(key)
            set_example_attributes(example, fields, val)

        return example

    def from_list(self, data):
        """
        Method creates example from data in list format.

        Parameters
        ----------
        data : list
            list containing values for fields in order that the fields were given to
            example factory

        Returns
        -------
        example : Example
            example instance with given data saved to fields
        """
        example = Example()

        for value, field in filter(lambda el: el[1] is not None, zip(data, self.fields)):
            set_example_attributes(example, field, value)

        return example

    def from_xml_str(self, data):
        """
        Method creates and Example from xml string.

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
            If the name is not contained in the xml string.
        ParseError
            If there was a problem while parsing xml sting, invalid xml.
        """
        example = Example()

        # we ignore columns with field mappings set to None
        items = filter(lambda el: el[1] is not None, self.fields.items())
        root = ET.fromstring(data)

        for name, field in items:
            node = root.find(name)

            if node is None:
                if root.tag == name:
                    node = root
                else:
                    raise ValueError(
                        f"Specified name {name} was not found in the input data"
                    )

            val = node.text
            set_example_attributes(example, field, val)

        return example

    def from_json(self, data):
        """
        Creates an Example from a JSON object and the corresponding fields.

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
            If JSON doesn't contain key name.
        """

        return self.from_dict(json.loads(data))

    def from_csv(self, data, field_to_index=None, delimiter=","):
        """
        Creates an Example from a CSV line and a corresponding list or dict of
        Fields.

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

    def from_fields_tree(self, data, subtrees=False, label_transform=None):
        """
        Creates an Example (or multiple Examples) from a string representing an
        nltk tree and a list of corresponding values.

        Parameters
        ----------
        data : str
            A string containing an nltk tree whose values are to be mapped
            to Fields.
        subtrees : bool
            A flag denoting whether an example will be created from every
            subtree in the tree (when set to True), or just from the whole
            tree (when set to False).
        label_transform : callable
            A function which converts the tree labels to a string representation,
            if wished. Useful for converting multiclass tasks to binary (SST) and
            making labels verbose. If None, the labels are not changed.

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
            subtree_lists = [tree_to_list(subtree) for subtree in tree.subtrees()]
            if label_transform is not None:
                # This is perhaps inefficient but probably the best place to insert this
                subtree_lists = [
                    [text, label_transform(label)] for text, label in subtree_lists
                ]
            # an example is created for each subtree
            return [self.from_list(subtree_list) for subtree_list in subtree_lists]
        else:
            text, label = tree_to_list(tree)
            if label_transform is not None:
                label = label_transform(label)
            return self.from_list([text, label])

    def from_format(self, data, format_tag: Union[ExampleFormat, str]):

        if isinstance(format_tag, ExampleFormat):
            format_str = format_tag.value

        elif isinstance(format_tag, str):
            format_str = format_tag.lower()

        else:
            raise TypeError(
                "format_tag must be either an ExampleFormat or a string. "
                f"Passed value is of type : '{type(format_tag).__name__}'"
            )

        factory_method = FACTORY_METHOD_DICT.get(format_str)
        if factory_method is None:
            raise ValueError(f"Unsupported example format: '{format_str}'")

        return factory_method(data, self)


def tree_to_list(tree):
    """
    Method joins tree leaves and label in one list.

    Parameters
    ----------
    tree : tree
        nltk tree instance

    Returns
    -------
    tree_list : list
        tree represented as list with its label
    """
    return [" ".join(tree.leaves()), tree.label()]


def set_example_attributes(example, field, val):
    """
    Method sets example attributes with given values.

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
            example[name] = data

    if isinstance(field, tuple):
        for f in field:
            set_example_attributes_for_single_field(example, f, val)

    else:
        set_example_attributes_for_single_field(example, field, val)

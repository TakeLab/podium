"""Example module that defines mapping for single data instance."""
import csv
import json
import logging
import xml.etree.ElementTree as ET

_LOGGER = logging.getLogger(__name__)


class Example(object):
    """Defines a single training or test example.

    An Example object (or multiple objects) is created by mapping each
    column to zero or more Field objects and setting those Fields as
    attributes of the Example.
    """

    @classmethod
    def from_json(cls, data, fields):
        """ Creates an Example from a JSON object and the
        corresponding fields.


        Parameters
        ----------
        data : str
            A string containing a single JSON object
            (key-value pairs surrounded by curly braces).
        fields : dict
            A dict mapping column names to Fields (or tuples of Fields).
            Columns that map to None will be ignored.

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

        return cls.from_dict(json.loads(data), fields)

    @classmethod
    def from_csv(cls, data, fields, field_to_index=None, delimiter=","):
        """ Creates an Example from a CSV line and a corresponding
        list or dict of Fields.

        Parameters
        ----------
        data : str
            A string containing a single row of values separated by the
            given delimiter.
        fields : (dict | list)
            Can be either a dict mapping column names to Fields
            (or tuples of Fields), or a list of Fields (or tuples of Fields).
            A Field value of None means the corresponding column will
            be ignored.
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

        if isinstance(fields, list):
            return cls.from_list(elements, fields)
        else:
            data_dict = {f: elements[idx] for f, idx in field_to_index.items()}
            return cls.from_dict(data_dict, fields)

    @classmethod
    def from_dict(cls, data, fields):
        """ Creates an Example from a dict of Fields and a dict of
        corresponding values.

        Parameters
        ----------
        data : dict
            A dict containing the values of a single row of data, that are
            to be mapped to Fields.
        fields : dict
            A dict mapping column names to Fields (or tuples of Fields).
            Columns that map to None will be ignored.

        Returns
        -------
        Example
            An Example whose attributes are the given Fields created with the
            given column values. These Fields can be accessed by their names.

        Raises
        ------
        ValueError
            if name mapping for a field is not present in dictionary
        """

        example = cls()

        # we ignore columns with field mappings set to None
        items = filter(lambda el: el[1] is not None, fields.items())
        for key, field in items:
            if key not in data:
                error_msg = f"Specified key {key} was not found "\
                            "in the input data"
                _LOGGER.error(error_msg)
                raise ValueError(error_msg)

            val = data[key]
            set_example_attributes(example, field, val)

        return example

    @classmethod
    def from_xml_str(cls, data, fields):
        """Method creates and Example from xml string.

        Parameters
        ----------
        data : str
            XML formated string that contains the values of a single data
            instance, that are to be mapped to Fields.
        fields : dict
            A dict mapping column names to Fields (or tuples of Fields).
            Columns that map to None will be ignored.

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
        example = cls()

        # we ignore columns with field mappings set to None
        items = filter(lambda el: el[1] is not None, fields.items())
        root = ET.fromstring(data)

        for name, field in items:
            node = root.find(name)

            if node is None:
                if root.tag == name:
                    node = root
                else:
                    error_msg = f"Specified name {name} was not found in the "\
                                "input data"
                    _LOGGER.error(error_msg)
                    raise ValueError(error_msg)

            val = node.text
            set_example_attributes(example, field, val)

        return example

    @classmethod
    def from_list(cls, data, fields):
        """ Creates an Example from a list of Fields and a list of
        corresponding values.

        Parameters
        ----------
        data : list
            A list containing the values of a single row of data, that are
            to be mapped to Fields.
        fields : list
            A list of Fields (or tuples of Fields). A None value means that
            the corresponding column will be ignored.

        Returns
        -------
        Example
            An Example whose attributes are the given Fields created with the
            given column values. These Fields can be accessed by their names.
        """

        example = cls()

        # we ignore columns with field mappings set to None
        data_fields = filter(lambda el: el[1] is not None, zip(data, fields))
        for val, field in data_fields:
            set_example_attributes(example, field, val)

        return example

    @classmethod
    def from_tree(cls, data, fields, subtrees=False):
        """ Creates an Example (or multiple Examples) from a string
        representing an nltk tree and a list of corresponding values.

        Parameters
        ----------
        data : str
            A string containing an nltk tree whose values are to be mapped
            to Fields.
        fields : list
            A list of Fields (or tuples of Fields). A None value means that
            the corresponding column will be ignored.
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
            return [cls.from_list(subtree_list, fields) for subtree_list in
                    subtree_lists]
        else:
            return cls.from_list(tree_to_list(tree), fields)


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
    if isinstance(field, tuple):
        for f in field:
            setattr(example, f.name, f.preprocess(val))
    else:
        setattr(example, field.name, field.preprocess(val))

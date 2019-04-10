import logging
import json

import xml.etree.ElementTree as ET

from recordclass import structclass
from uuid import uuid4


_LOGGER = logging.getLogger(__name__)


class ExampleFactory:

    def __init__(self, fields):

        if isinstance(fields, dict):
            self.fields = {val_name: fields_
                           for val_name, fields_
                           in fields.items()
                           if fields_ is not None}
        else:
            self.fields = fields

        fieldnames = tuple(field.name for field in unpack_fields(fields))
        uid = uuid4()
        example_class_name = "Example_class_" + str(uid).replace("-", "_")
        self.example_factory = structclass(example_class_name,
                                           fieldnames,
                                           defaults=[None] * len(fieldnames)
                                           )
        globals()[example_class_name] = self.example_factory

    def from_dict(self, data):
        example = self.example_factory()

        for key, fields in self.fields.items():
            val = data.get(key)
            _set_example_attributes(example, fields, val)

        return example

    def from_list(self, data):
        example = self.example_factory()
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
        example = self.example_factory()

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

        return self.from_dict(json.loads(data))


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

from recordclass import recordclass
import xml.etree.ElementTree as ET
import logging

from takepod.storage.dataset import unpack_fields

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
        self.example_factory = recordclass("Example",
                                           fieldnames,
                                           defaults=[None] * len(fieldnames)
                                           )

    def from_dict(self, data):
        example = self.example_factory()

        for key, fields in self.fields.items():
            val = data.get(key)
            set_example_attributes(example, fields, val)

        return example

    def from_list(self, data):
        example = self.example_factory()
        for value, field in filter(lambda el: el[1] is not None, zip(data, self.fields)):
            set_example_attributes(example, field, value)


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
                    error_msg = f"Specified name {name} was not found in the "\
                                "input data"
                    _LOGGER.error(error_msg)
                    raise ValueError(error_msg)

            val = node.text
            set_example_attributes(example, field, val)

        return example


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

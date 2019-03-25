from recordclass import recordclass

from takepod.storage.dataset import unpack_fields
from takepod.storage.example import Example


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

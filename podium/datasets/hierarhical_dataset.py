import json
from dataclasses import dataclass
from typing import Optional, Tuple

from podium.datasets.dataset import Dataset
from podium.field import unpack_fields

from .example_factory import Example, ExampleFactory


@dataclass
class Node:
    """
    Class defines a node in hierarhical dataset.

    Attributes
        ----------
        example : Example
            example instance containing node data
        index : int
            index in current hierarchy level
        parent : Node
            parent node
        children : tuple(Node), optional
            children nodes
    """

    example: Example
    index: int
    parent: "Node"
    children: Optional[Tuple["Node"]] = None


class HierarchicalDataset:
    """
    Container for datasets with a hierarchical structure of examples which have
    the same structure on every level of the hierarchy.
    """

    def __init__(self, parser, fields):
        """
        Constructs the Hierarchical dataset.

        Parameters
        ----------
        parser : callable
            Callable taking (raw_example, fields, depth) and returning a tuple containing
            (example, raw_children).
            Arguments:
                Raw_example: a dict representation of the
                    example.

                Fields: a dict mapping keys in the raw_example  to corresponding
                    fields in the dataset.

                Depth: an int marking the depth of the current
                    example in the hierarchy.

            Return values are:
                Example: Example instance containing the data in raw_example.

                Raw_children: iterable of dicts representing the children of raw_example


        fields : dict(str, Field)
            Dict mapping keys in the raw_example dict to their corresponding fields.
        """
        self.fields = unpack_fields(fields)
        self.field_dict = {field.name: field for field in self.fields}
        self._example_factory = ExampleFactory(fields)
        self._parser = parser
        self._size = 0
        self._max_depth = 0

    @staticmethod
    def from_json(dataset, fields, parser):
        """
        Makes an HierarchicalDataset from a JSON formatted string.

        Parameters
        ----------
        dataset : str
            Dataset in JSON format. The root element of the JSON string must be
            a list of root examples.

        fields : dict(str, Field)
            a dict mapping keys in the raw_example to corresponding
            fields in the dataset.

        parser : callable(raw_example, fields, depth) returning (example, raw_children)
            Callable taking (raw_example, fields, depth) and returning a tuple containing
            (example, raw_children).

        Returns
        -------
            HierarchicalDataset
                dataset containing the data

        Raises
        ------
            If the base element in the JSON string is not a list of root elements.
        """
        ds = HierarchicalDataset(parser, fields)

        root_examples = json.loads(dataset)
        if not isinstance(root_examples, list):
            raise ValueError(
                "The base element in the JSON string must be a list of root elements."
            )

        ds._load(root_examples)

        return ds

    @staticmethod
    def get_default_dict_parser(child_attribute_name):
        """
        Returns a callable instance that can be used for parsing datasets in
        which examples on all levels in the hierarchy have children under the
        same key.

        Parameters
        ----------
        child_attribute_name : str
            key used for accessing children in the examples

        Returns
        -------
            Callable(raw_example, fields, depth) returning (example, raw_children).
        """

        def default_dict_parser(raw_example, example_factory, depth):
            example = example_factory.from_dict(raw_example)
            children = raw_example.get(child_attribute_name, ())
            return example, children

        return default_dict_parser

    def _load(self, root_examples):
        """
        Starts the parsing of the dataset.

        Parameters
        ----------
        root_examples : iterable(dict(str, object))
            iterable containing the root examples in raw dict form.
        """
        self._root_nodes = tuple(self._parse(root, None, 0) for root in root_examples)

    def finalize_fields(self):
        """
        Finalizes all fields in this dataset.
        """

        for field in self.fields:
            field.finalize()

    def _parse(self, raw_object, parent, depth):
        """
        Parses an raw example.

        Parameters
        ----------
        raw_object : dict(str, object)
            Example in raw dict form.

        parent
            Parent node of the example to be parsed. None for root nodes.

        depth
            Depth of the example to be parsed in the hierarchy. Depth of root nodes is 0.

        Returns
        -------
        Node
            Node parsed from the raw example.
        """
        example, raw_children = self._parser(raw_object, self._example_factory, depth)

        index = self._size
        self._size += 1

        current_node = Node(example, index, parent)
        children = tuple(self._parse(c, current_node, depth + 1) for c in raw_children)
        current_node.children = children

        self._max_depth = max(self._max_depth, depth)

        return current_node

    def _node_iterator(self):
        def flat_node_iterator(node):
            yield node
            for subnode in node.children:
                for ex in flat_node_iterator(subnode):
                    yield ex

        for root_node in self._root_nodes:
            for ex in flat_node_iterator(root_node):
                yield ex

    def flatten(self):
        """
        Returns an iterable iterating trough examples in the dataset as if it
        was a standard Dataset. The iteration is done in pre-order.

        Returns
        -------
        iterable
             iterable iterating through examples in the dataset.
        """
        for node in self._node_iterator():
            yield node.example

    def as_flat_dataset(self):
        """
        Returns a standard Dataset containing the examples in order as defined
        in 'flatten'.

        Returns
        -------
        Dataset
            a standard Dataset
        """
        return Dataset(list(self.flatten()), self.field_dict)

    @property
    def depth(self):
        """
        Returns
        -------
        int
            the maximum depth of a node in the hierarchy.
        """
        return self._max_depth

    def _get_node_by_index(self, index):
        """
        Returns the node with the provided index.

        Parameters
        ----------
        index : int
            Index of the node to be fetched.

        Returns
        -------
        Node
            the node with the provided index.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of bounds. Must be within [0, len(dataset) - 1]"
            )

        def get_item(nodes, index):
            """
            Right bisect binary search.

            Parameters
            ----------
            nodes : list(Node)
                Nodes to be searched.

            index : int
                index of the node to fetch.

            Returns
            -------
            Node
                the node with the provided index.
            """
            start = 0
            end = len(nodes)

            while start < end:
                middle = (start + end) // 2
                middle_index = nodes[middle].index

                if index < middle_index:
                    end = middle

                else:
                    start = middle + 1

            if nodes[start - 1].index == index:
                return nodes[start - 1]

            else:
                return get_item(nodes[start - 1].children, index)

        return get_item(self._root_nodes, index)

    @staticmethod
    def _get_node_context(node, levels=None):
        """
        Returns an Iterator iterating through the context of the passed node.

        Parameters
        ----------
        node : Node
            Node for which the context should be retrieved.
        levels : the maximum number of levels of the hierarchy the context should contain.
            If None, the context will contain all levels up to the root nodes of the
            dataset.

        Returns
        -------
        Iterator(Node)
            an Iterator iterating through the context of the passed node
        """
        levels = float("Inf") if levels is None else levels
        if levels < 0:
            raise ValueError(
                "Number of context levels must be greater or equal to 0. "
                f"Passed value: {levels}"
            )

        parent = node
        while parent.parent is not None and levels >= 0:
            parent = parent.parent
            levels -= 1

        def context_iterator(start_node, finish_node):
            if start_node is finish_node:
                return

            yield start_node.example

            children = start_node.children
            i = 0
            while True:
                if i == len(children) - 1 or children[i + 1].index > finish_node.index:
                    for sub_child in context_iterator(children[i], finish_node):
                        yield sub_child

                    return

                else:
                    yield children[i].example
                    i += 1

        return context_iterator(parent, node)

    def get_context(self, index, levels=None):
        """
        Returns an Iterator iterating through the context of the Example with
        the passed index.

        Parameters
        ----------
        index : int
            Index of the Example the context should be retrieved for.
        levels : int
            the maximum number of levels of the hierarchy the context should contain.
            If None, the context will contain all levels up to the root node of the
            dataset.

        Returns
        -------
        Iterator(Node)
            an Iterator iterating through the context of the Example with the passed
            index.

        Raises
        ------
            If levels is less than 0.
        """
        node = self._get_node_by_index(index)
        return HierarchicalDataset._get_node_context(node, levels)

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        return self._get_node_by_index(index).example

    def __getstate__(self):
        """
        Method obtains dataset state. It is used for pickling dataset data to
        file.

        Returns
        -------
        state : dict
            dataset state dictionary
        """
        state = self.__dict__.copy()
        del state["_parser"]
        return state

    def __setstate__(self, state):
        """
        Method sets dataset state. It is used for unpickling dataset data from
        file.

        Parameters
        ----------
        state : dict
            dataset state dictionary
        """
        self.__dict__ = state

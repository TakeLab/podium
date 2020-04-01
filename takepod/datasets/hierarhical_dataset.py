import json
import logging
from itertools import chain

from takepod.datasets.dataset import Dataset
from takepod.storage.field import unpack_fields
from takepod.storage.example_factory import ExampleFactory

_LOGGER = logging.getLogger(__name__)


class HierarchicalDataset:
    """Container for datasets with a hierarchical structure of examples which have the
    same structure on every level of the hierarchy.
    """

    class Node(object):
        """Class defines a node in hierarchical dataset.

        Attributes
        ----------
        example : Example
            example instance containing node data
        index : int
            index in current hierarchy level
        parent_node : Node
            parent node
        next_node : Node
            next node at the same level
        children_nodes : tuple(Node)
            children nodes
        """
        __slots__ = 'example', 'index', 'parent_node', 'next_node', 'children_nodes'

        def __init__(self, example, index, parent_node,
                     next_node=None, children_nodes=None):
            self.example = example
            self.index = index
            self.parent_node = parent_node
            self.next_node = next_node
            self.children_nodes = children_nodes

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

        """
        ds = HierarchicalDataset(parser, fields)

        root_examples = json.loads(dataset)
        if not isinstance(root_examples, list):
            error_msg = "The base element in the JSON string must be a list " \
                        "of root elements."
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)

        ds._load(root_examples)

        return ds

    @staticmethod
    def get_default_dict_parser(child_attribute_name):
        """Returns a callable instance that can be used for parsing datasets in which
        examples on all levels in the hierarchy have children under the same key.

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
        """Starts the parsing of the dataset.

        Parameters
        ----------
        root_examples : iterable(dict(str, object))
            iterable containing the root examples in raw dict form.

        """
        self._root_nodes = tuple(self._parse(root, None, 0) for root in root_examples)
        for root, next_root in zip(self._root_nodes, self._root_nodes[1:]):
            root.next_node = next_root

    def finalize_fields(self):
        """Finalizes all fields in this dataset."""

        fields_to_build = [f for f in self.fields if
                           not f.eager and f.use_vocab]

        if fields_to_build:
            for example in self.flatten():
                for field in fields_to_build:
                    field.update_vocab(*getattr(example, field.name))

        for field in self.fields:
            field.finalize()

    def _parse(self, raw_object, parent, depth):
        """Parses an raw example.

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

        current_node = HierarchicalDataset.Node(example, index, parent)
        children = tuple(self._parse(c, current_node, depth + 1) for c in raw_children)
        for child, next_child in zip(children, children[1:]):
            child.next_node = next_child
        current_node.children_nodes = children

        self._max_depth = max(self._max_depth, depth)

        return current_node

    def _node_iterator(self):

        def flat_node_iterator(node):
            yield node
            for subnode in node.children_nodes:
                for ex in flat_node_iterator(subnode):
                    yield ex

        for root_node in self._root_nodes:
            for ex in flat_node_iterator(root_node):
                yield ex

    def flatten(self):
        """
        Returns an iterable iterating trough examples in the dataset as if it was a
        standard Dataset. The iteration is done in pre-order.

        Returns
        -------
        iterable
             iterable iterating through examples in the dataset.
        """
        for node in self._node_iterator():
            yield node.example

    def as_flat_dataset(self):
        """Returns a standard Dataset containing the examples
        in order as defined in 'flatten'.

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
        """Returns the node with the provided index.

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
            if the index is out of bounds.

        """
        if index < 0 or index >= len(self):
            error_msg = "Index {} out of bounds. Must be within " \
                        "[0, len(dataset) - 1]".format(index)
            _LOGGER.error(error_msg)
            raise IndexError(error_msg)

        def get_item(nodes, index):
            """Right bisect binary search.

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
                return get_item(nodes[start - 1].children_nodes, index)

        return get_item(self._root_nodes, index)

    @staticmethod
    def _get_pre_context_nodes(node, levels=None):
        """Returns an Iterator iterating through the pre-context Nodes of the passed node.

        Parameters
        ----------
        node : Node
            Node for which the context should be retrieved.
        levels :
            the maximum number of levels of the hierarchy above the node the context
            should contain.

            If 0, the parent Node and all its children nodes before the passed node will
            will be contained in the context.

            If None, the context will contain all levels up to the root node of the
            dataset.

        Returns
        -------
        Iterator(Node)
            an Iterator iterating through the context of the passed node

        """
        levels = float('Inf') if levels is None else levels
        if levels < 0:
            error_msg = "Number of context levels must be greater or equal to 0." \
                        " Passed value: {}".format(levels)
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)

        # Go up the hierarchy `level` times
        parent = node
        while parent.parent_node is not None and levels >= 0:
            parent = parent.parent_node
            levels -= 1

        def pre_context_node_iterator(start_node, finish_node):
            if start_node is finish_node:
                return

            # Return root node
            yield start_node

            children = start_node.children_nodes
            i = 0
            # Iterate over children
            while True:
                if i == len(children) - 1 or children[i + 1].index > finish_node.index:
                    for sub_child in pre_context_node_iterator(children[i], finish_node):
                        yield sub_child

                    return

                else:
                    yield children[i]
                    i += 1

        return pre_context_node_iterator(parent, node)

    def get_pre_context_examples(self, index, levels=None):
        """Returns an Iterator iterating over the pre-context of the Example with the
        passed index.

        Parameters
        ----------
        index : int
            Index of the Example the pre-context should be retrieved for.
        levels :
            the maximum number of levels of the hierarchy above the node the context
            should contain.

            If 0, the parent Node and all its children nodes before the passed node will
            will be contained in the context.

            If None, the context will contain all levels up to the root node of the
            dataset.

        Returns
        -------
        Iterator(Example)
            an Iterator iterating through the pre-context of the Example with the passed
            index.

        """
        node = self._get_node_by_index(index)
        pre_context_nodes = HierarchicalDataset._get_pre_context_nodes(node, levels)
        return (n.example for n in pre_context_nodes)

    @staticmethod
    def _get_post_context_nodes(node: Node,
                                levels: int = 1,
                                skip_same_level=False,
                                skip_same_level_if_root=True):
        levels = float('Inf') if levels is None else levels
        if levels < 0:
            error_msg = "Number of context levels must be greater or equal to 0." \
                        " Passed value: {}".format(levels)
            _LOGGER.error(error_msg)
            raise ValueError(error_msg)

        def children_iterator(parent_node, current_level, max_level):
            if current_level >= max_level:
                return

            for child in parent_node.children_nodes:
                yield child
                if current_level < max_level - 1:
                    for sub_child in children_iterator(child,
                                                       current_level + 1,
                                                       max_level):
                        yield sub_child

        def same_level_iterator(node):
            node = node.next_node
            while node is not None:
                yield node
                node = node.next_node

        children = children_iterator(node, 0, levels)

        if skip_same_level \
                or skip_same_level_if_root and node.parent_node is None:
            return children

        else:
            same_level_nodes = same_level_iterator(node)
            return chain(children, same_level_nodes)

    def get_post_context_examples(self,
                                  index: int,
                                  levels: int = 1,
                                  skip_same_level=False,
                                  skip_same_level_if_root=True):
        """Returns an Iterator iterating over the post-context of the Example with the
            passed index. The post-context contains `levels` levels of children of the
            indexed Example and all Examples that come after the indexed example at the
            same level.

            Parameters
            ----------
            index : int
                Index of the Example the post-context should be retrieved for.
            levels : int
                the maximum number of levels of children below the node the
                context should contain. For example, for `levels` 2 the context will
                contain the children, and the children of those children.

                If 0, no children will be contained in the context.

                If None, the context will contain all children.
            skip_same_level: bool
                If True, the Examples that come after the indexed Example at the same
                level will not be added to the context.
            skip_same_level_if_root: bool
                If True, the Examples that come after the indexed Example at the same
                level will not be added to the context if the indexed Example is a root
                node.

            Returns
            -------
            Iterator(Example)
                an Iterator iterating through the pre-context of the Example with the
                passed index.
        """
        node = self._get_node_by_index(index)
        post_context_nodes = HierarchicalDataset._get_post_context_nodes(
            node,
            levels,
            skip_same_level,
            skip_same_level_if_root
        )
        return (n.example for n in post_context_nodes)

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        return self._get_node_by_index(index).example

    def __getstate__(self):
        """Method obtains dataset state. It is used for pickling dataset data
        to file.

        Returns
        -------
        state : dict
            dataset state dictionary
        """
        d = dict(self.__dict__)

        del d["_parser"]

        return d

    def __setstate__(self, state):
        """Method sets dataset state. It is used for unpickling dataset data
        from file.

        Parameters
        ----------
        state : dict
            dataset state dictionary
        """
        self.__dict__ = state

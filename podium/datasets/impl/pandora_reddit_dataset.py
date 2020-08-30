import csv
from collections import defaultdict

from podium.datasets import HierarchicalDataset
from podium.storage import Field


class PandoraDataset(HierarchicalDataset):

    @staticmethod
    def get_dataset(comments_path, fields=None):

        if fields is None:
            fields = PandoraDataset.get_default_fields()

        top_level_comments = list()  # list of top level comments
        all_comments = list()  # list of all comments
        children = defaultdict(list)  # dict mapping id -> children
        loaded_comments = 0
        children_attached = 0

        with open(comments_path, "r") as comments_file:
            comments_reader = csv.DictReader(comments_file)

            for comment in comments_reader:
                loaded_comments += 1
                prefix, parent_id = comment['parent_id'].split('_', 1)
                all_comments.append(comment)

                if prefix == 't1':  # t1 denotes a child of another comment
                    children[parent_id].append(comment)

                elif prefix == 't3':  # t3 denotes a top-level comment
                    top_level_comments.append(comment)

                else:
                    raise Exception("Unrecognised parent_id prefix : {}".format(prefix))

        for comment in all_comments:
            comments_children = children.get(comment['id'], ())
            children_attached += len(comments_children)
            comment['children'] = comments_children

        assert loaded_comments == len(top_level_comments) + children_attached, \
            "Not all comments have parents, can't build treeah "

        return PandoraDataset.from_dicts(top_level_comments, fields, 'children')

    @staticmethod
    def get_default_fields():
        # TODO add real fields
        id_field = Field('id', tokenize=False, store_as_raw=True,
                         is_numericalizable=False)

        return {
            'id': id_field
        }

if __name__ == '__main__':
    dataset = PandoraDataset.get_dataset("/home/ivans/reddit_dataset/all_comments_since_2015.csv")
    i = 1
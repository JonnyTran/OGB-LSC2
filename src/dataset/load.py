from argparse import ArgumentParser, Namespace
from unittest import mock

from torch_geometric.data import InMemoryDataset


def load_dataset(args: Namespace) -> InMemoryDataset:
    if args.platform == "pyg":
        from ogb.graphproppred import PygGraphPropPredDataset as GraphPropPredDataset
        from ogb.nodeproppred import PygNodePropPredDataset as NodePropPredDataset
        from ogb.linkproppred import PygLinkPropPredDataset as LinkPropPredDataset
    elif args.platform == "dgl":
        from ogb.graphproppred import DglGraphPropPredDataset as GraphPropPredDataset
        from ogb.nodeproppred import DglNodePropPredDataset as NodePropPredDataset
        from ogb.linkproppred import DglLinkPropPredDataset as LinkPropPredDataset

    # Avoid running into prompts while building docker image
    with mock.patch("ogb.lsc.input", new=fake_input("y")), \
            mock.patch("ogb.utils.url.input", new=fake_input("y")), \
            mock.patch("ogb.nodeproppred.input", new=fake_input("y")):

        # LSC Datasets
        if args.dataset == "MAG240M":
            from ogb.lsc import MAG240MDataset
            dataset = MAG240MDataset(root=args.root)
        elif args.dataset == "WikiKG90Mv2":
            from ogb.lsc import WikiKG90Mv2Dataset
            dataset = WikiKG90Mv2Dataset(root=args.root)
        elif args.dataset == "PCQM4Mv2":
            if args.platform == "pyg":
                from ogb.lsc.pcqm4mv2_pyg import PCQM4Mv2Dataset
            else:
                from ogb.lsc.pcqm4mv2_dgl import PCQM4Mv2Dataset
            dataset = PCQM4Mv2Dataset(root=args.root)

        elif "ogbn" in args.dataset:
            dataset = NodePropPredDataset(name=args.dataset, root=args.root)
        elif "ogbl" in args.dataset:
            dataset = LinkPropPredDataset(name=args.dataset, root=args.root)
        elif "ogbg" in args.dataset:
            dataset = GraphPropPredDataset(name=args.dataset, root=args.root)

    return dataset


def fake_input(default_response=None):
    '''Creates a replacement for the `input` function that will
    return a default response if one was provided.'''

    def _input(prompt):
        return default_response if default_response else input(prompt)

    return _input


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default="MAG240M", required=True)
    parser.add_argument('-r', '--root', type=str, default="~/dataset/")
    parser.add_argument('-p', '--platform', type=str, default="pyg", required=False)
    args = parser.parse_args()

    load_dataset(args)

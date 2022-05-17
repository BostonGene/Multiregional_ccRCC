import warnings

import numpy as np
import pandas as pd


class GeneSet(object):
    def __init__(self, name, descr, genes):
        self.name = name
        self.descr = descr
        self.genes = set(genes)
        self.genes_ordered = list(genes)

    def __str__(self):
        return '{}\t{}\t{}'.format(self.name, self.descr, '\t'.join(self.genes))


def read_gene_sets(gmt_file):
    """
    Return dict {geneset_name : GeneSet object}

    :param gmt_file: str, path to .gmt file
    :return: dict
    """
    gene_sets = {}
    with open(gmt_file) as handle:
        for line in handle:
            items = line.strip().split('\t')
            name = items[0].strip()
            description = items[1].strip()
            genes = set([gene.strip() for gene in items[2:]])
            gene_sets[name] = GeneSet(name, description, genes)

    return gene_sets


def median_scale(data, clip=None):
    c_data = (data - data.median()) / data.mad()
    if clip is not None:
        return c_data.clip(-clip, clip)
    return c_data


def read_dataset(file, sep='\t', header=0, index_col=0, comment=None):
    return pd.read_csv(file, sep=sep, header=header, index_col=index_col,
                       na_values=['Na', 'NA', 'NAN'], comment=comment)


def item_series(item, indexed=None):
    """
    Creates a series filled with item with indexes from indexed (if Series-like) or numerical indexes (size=indexed)
    :param item: value for filling
    :param indexed:
    :return:
    """
    if indexed is not None:
        if hasattr(indexed, 'index'):
            return pd.Series([item] * len(indexed), index=indexed.index)
        elif type(indexed) is int and indexed > 0:
            return pd.Series([item] * indexed, index=np.arange(indexed))
    return pd.Series()


def to_common_samples(df_list=()):
    """
    Accepts a list of dataframes. Returns all dataframes with only intersecting indexes
    :param df_list: list of pd.DataFrame
    :return: pd.DataFrame
    """
    cs = set(df_list[0].index)
    for i in range(1, len(df_list)):
        cs = cs.intersection(df_list[i].index)

    if len(cs) < 1:
        warnings.warn('No common samples!')
    return [df_list[i].loc[list(cs)] for i in range(len(df_list))]


def pivot_vectors(vec1, vec2):
    """
    Aggregates 2 vectors into a table with amount of pairs (vec1.x, vec2.y) in a cell
    Both series must have same index.
    Else different indexes values will be counted in a_label_1/na_label_2 columns if specified or ignored
    :param vec1: pd.Series
    :param vec2: pd.Series
    :return: pivot table
    """

    name1 = str(vec1.name)
    if vec1.name is None:
        name1 = 'V1'

    name2 = str(vec2.name)
    if vec2.name is None:
        name2 = 'V2'

    sub_df = pd.DataFrame({name1: vec1,
                           name2: vec2})

    sub_df = sub_df.assign(N=item_series(1, sub_df))

    return pd.pivot_table(data=sub_df, columns=name1,
                          index=name2, values='N', aggfunc=sum).fillna(0).astype(int)


def normalize(data):
    return data.div(data.sum())


def to_linear_ranges(data, ps=[0.5]):
    """
    Annotates samples of a numeric series by its range in linear form. 
    Change amount of groups and thresholds by modifying ps arg
    :param data: pd.Series with numeric-like data
    :param ps: percentiles marks to assosiate sumples
    :return:
    """
    ann = []
    tr = 0
    
    data_range = data.max() - data.min()

    for tr2 in list(np.sort(ps)) + [1]:
        if tr == 0:
            x = list(data[np.logical_and(data >= (data.min() + data_range * tr), 
                                         data <= (data.min() + data_range * tr2))].index)
        else:
            x = list(data[np.logical_and(data > (data.min() + data_range * tr), 
                                         data <= (data.min() + data_range * tr2))].index)

        ann.append(pd.Series(['{}p<x<{}p'.format(tr, tr2)]*len(x), index=x))
        tr = tr2
    
    return pd.concat(ann)




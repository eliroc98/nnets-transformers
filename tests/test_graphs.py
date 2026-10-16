import unittest
import pandas as pd
import networkx as nx
from nnets_transformer.analysis.graphs import (
    Component, build_graph, centralities, CORRECTED, LEGACY_NOTEBOOK, LEGACY_RENDERER,
)


class GraphContracts(unittest.TestCase):
    def setUp(self):
        self.names = ["hook_embed", "blocks.0.attn.hook_q", "blocks.0.attn.hook_q.0",
                      "blocks.0.attn.hook_rot_q", "blocks.0.attn.hook_rot_q.0",
                      "blocks.0.attn.hook_z", "blocks.0.attn.hook_z.0",
                      "blocks.0.mlp.hook_post", "blocks.1.mlp.hook_post", "logit"]
        self.inventory = [Component.parse(n, 2) for n in self.names]
        self.frame = pd.DataFrame(1., index=self.names, columns=self.names)
        self.frame.loc["hook_embed", "blocks.0.attn.hook_q"] = .2
        self.frame.loc["blocks.0.attn.hook_q", "blocks.0.attn.hook_rot_q"] = .3
        self.frame.loc["blocks.0.attn.hook_rot_q", "blocks.0.attn.hook_z"] = .4
        self.frame.loc["blocks.0.attn.hook_z", "blocks.0.mlp.hook_post"] = .1

    def graph(self, version=CORRECTED, **kwargs):
        return build_graph(self.frame, inventory=self.inventory, threshold=.7, version=version,
                           resolution=kwargs.pop("resolution", "component"), **kwargs)

    def test_legacy_double_transform_and_active_nodes(self):
        a, b = self.graph(LEGACY_NOTEBOOK), self.graph(LEGACY_RENDERER)
        self.assertEqual(set(a), {"emb", "attn.q.0"})
        self.assertEqual(list(a.edges), list(b.edges))
        self.assertAlmostEqual(a["emb"]["attn.q.0"]["weight"], .8)
        self.assertAlmostEqual(b["emb"]["attn.q.0"]["weight"], .2)

    def test_corrected_inventory_ordering_distance(self):
        graph = self.graph()
        self.assertEqual(len(graph), 7)
        self.assertEqual(graph.number_of_edges(), 3)
        self.assertNotIn(("blocks.0.attn.hook_z", "blocks.0.mlp.hook_post"), graph.edges)
        attrs = graph["hook_embed"]["blocks.0.attn.hook_q"]
        self.assertAlmostEqual(attrs["strength"], .8)
        self.assertAlmostEqual(attrs["distance"], 1 / (.8 + 1e-8))
        self.assertEqual(graph.graph["active_nodes"], 4)
        self.assertTrue(nx.is_directed_acyclic_graph(graph))
        self.assertEqual(self.graph(architecture="sequential").number_of_edges(), 4)

    def test_head_resolution_includes_unobserved_isolates(self):
        graph = self.graph(resolution="head")
        self.assertIn("blocks.0.attn.hook_rot_q.0", graph)
        self.assertNotIn("blocks.0.attn.hook_rot_q", graph)
        self.assertIn("blocks.0.mlp.hook_post", graph)
        self.assertEqual(graph.degree("blocks.0.attn.hook_rot_q.0"), 0)

    def test_directed_path_metrics(self):
        graph = self.graph()
        metrics = centralities(graph)
        self.assertEqual(metrics["closeness_in"]["hook_embed"], 0)
        self.assertGreater(metrics["closeness_out"]["hook_embed"], 0)
        self.assertAlmostEqual(metrics["in_strength"]["blocks.0.attn.hook_q"], .8)

    def test_strict_threshold_and_nan(self):
        self.frame.loc["hook_embed", "blocks.0.attn.hook_q"] = .7
        self.frame.loc["blocks.0.attn.hook_q", "blocks.0.attn.hook_rot_q"] = float("nan")
        self.assertEqual(self.graph().number_of_edges(), 1)

    def test_invalid_and_unknown_definitions(self):
        with self.assertRaises(ValueError): self.graph("unversioned")
        with self.assertRaises(ValueError): self.graph(resolution="mixed")
        self.frame.loc["hook_embed", "blocks.0.attn.hook_q"] = 2
        with self.assertRaises(ValueError): self.graph()
        with self.assertRaises(ValueError):
            build_graph(self.frame, inventory=self.inventory[:-1], threshold=.7, version=CORRECTED)


if __name__ == "__main__":
    unittest.main()

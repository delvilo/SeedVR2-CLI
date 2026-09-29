import unittest
from omegaconf import OmegaConf, DictConfig, ListConfig
from src.common.config import resolve_recursive, resolve_inheritance

class ConfigTests(unittest.TestCase):
    def test_resolve_recursive_dict_and_list(self):
        config = OmegaConf.create({
            "a": 1,
            "b": [10, {"c": 20}, [30, 40]],
            "d": {"e": 50}
        })

        # Test custom resolver function
        def mock_resolver(cfg):
            if isinstance(cfg, DictConfig) and "a" in cfg:
                cfg["a"] = 100
            return cfg

        resolved = resolve_recursive(config, mock_resolver)
        self.assertEqual(resolved.a, 100)
        self.assertEqual(resolved.b[1].c, 20)
        self.assertEqual(resolved.b[2][0], 30)

    def test_resolve_recursive_inheritance(self):
        config = OmegaConf.create({
            "param": "value",
            "list_data": [{"item_id": 1}, {"item_id": 2}]
        })
        resolved = resolve_recursive(config, resolve_inheritance)
        self.assertEqual(resolved.param, "value")
        self.assertEqual(len(resolved.list_data), 2)

if __name__ == "__main__":
    unittest.main()

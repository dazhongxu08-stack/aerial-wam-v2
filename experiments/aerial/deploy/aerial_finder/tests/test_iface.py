from aerial_finder.iface.schema import ParseOut
from aerial_finder.iface.adapters import ParserAdapter
from aerial_finder.mission.parser import parse_instruction


def test_parse_out_hides_internal_keys():
    spec = parse_instruction("正前方10米找一个红色的凳子")
    out = ParseOut.from_spec_dict(spec.to_dict())
    d = out.to_dict()
    assert "label" not in d
    assert "distance_m" not in d
    assert d["object"] == "stool"
    assert d["color"] == "red"
    assert d["range_m"] == 10.0
    assert d["bearing_rad"] == 0.0


def test_parser_adapter():
    ad = ParserAdapter(parse_instruction)
    from aerial_finder.iface.schema import ParseIn
    d = ad.parse(ParseIn("前方3米蓝色桶")).to_dict()
    assert d["object"] == "barrel"
    assert d["color"] == "blue"
    assert d["range_m"] == 3.0

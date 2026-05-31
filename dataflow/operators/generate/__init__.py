from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cskg_triple_extractor import CSKGTripleExtraction
    from .finkg_4tuple_extractor import FinKGTupleExtraction
    from .geokg_4tuple_extractor import GeoKGTupleExtraction
    from .geokg_event_extractor import GeoKGEventExtraction
    from .hrkg_rel_triple_extractor import HRKGTripleExtraction
    from .kg_entity_extractor import KGEntityExtraction
    from .kg_triple_extractor import KGTripleExtraction
    from .kg_triple_merge import KGTripleMerger
    from .kg_tuple2text import KGTupleTextGeneration
    from .legalkg_triple_extractor import LegalKGTupleExtraction
    from .medkg_triple_extractor import MedKGTripleExtraction
    from .tkg_4tuple_extractor import TKGTupleExtraction
    from .tkg_4tuple_merge import TKGTupleMerger

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/generate/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, cur_path, _import_structure)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .geokg_entity_link2database import GeoKGEntityLink
    from .geokg_rel_4tuple_inference import GeoKGRelationInference
    from .kg_entity_alignment import KGGraphEntityAligner
    from .kg_entity_classification import KGEntityClassification
    from .kg_entity_disambiguation import KGEntityDisambiguation
    from .kg_entity_link2database import KGEntityLink
    from .kg_entity_normalization import KGEntityNormalization
    from .kg_triple_disambiguation import KGTripleDisambiguation
    from .kg_tuple_normalization import KGTupleNormalization
    from .tkg_4tuple_disambiguation import TKGTupleDisambiguation

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/refine/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, cur_path, _import_structure)

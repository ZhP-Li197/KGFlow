from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cskg_triple_adaptability_filtering import CSKGTripleAdapbilityFilter
    from .cskg_triple_rationale_filtering import CSKGTripleRationaleFilter
    from .domain_kg_tuple_ontology_filtering import GeoKGTupleFilter
    from .geokg_event_consistence_filtering import GeoKGEventConsistenceFilter
    from .geokg_event_location_filtering import GeoKGEventTupleLocationFilter
    from .geokg_event_rationale_filtering import GeoKGEventRationaleFilter
    from .geokg_event_time_filtering import GeoKGEventTupleTimeFilter
    from .hrkg_rel_triple_attri_filtering import HRKGRelationTripleAttributeFilter
    from .hrkg_rel_triple_completeness_filtering import HRKGTripleCompletenessFilter
    from .hrkg_rel_triple_consistency_filtering import HRKGTripleConsistenceFilter
    from .kg_entity_validation import KGEntityValidity
    from .kg_rel_triple_strength_filtering import KGTripleStrengthFilter
    from .kg_tuple_remove_repeated import KGTupleRemoveRepeated
    from .kg_tuple_validation import KGTupleValidity
    from .tkg_4tuple_time_sampling import TKGTupleTimeFilter

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/filter/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, cur_path, _import_structure)

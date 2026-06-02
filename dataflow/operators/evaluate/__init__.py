from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cskg_triple_adaptability_eval import CSKGTripleAdaptabilityEvaluator
    from .cskg_triple_rationale_eval import CSKGTripleRationaleEvaluator
    from .geokg_event_consistence_eval import GeoKGEventConsistenceEvaluator
    from .geokg_event_rationale_eval import GeoKGEventRationaleEvaluator
    from .geokg_event_summary import GeoKGTupleAttributeFrequencyEvaluator
    from .hrkg_rel_triple_attri_summary import HRKGTupleAttributeFrequencyEvaluator
    from .hrkg_rel_triple_completeness_eval import HRKGTripleCompletenessEvaluator
    from .hrkg_rel_triple_consistency_eval import HRKGTripleConsistencyEvaluator
    from .kg_rel_triple_consistency_eval import KGRelationTripleConsistencyEvaluator
    from .kg_rel_triple_nx_visual import KGRelationTripleVisualization
    from .kg_rel_triple_strength_eval import KGRelationStrengthScoring
    from .kg_rel_triple_topology_eval import KGRelationTripleTopologyEvaluator
    from .tkg_4tuple_time_summary import TKGTemporalStatistics
    from .kg_subgraph_connectivity_eval import KGSubgraphConnectivityEvaluator
    from .kg_subgraph_consistence_eval import KGSubgraphConsistency
    from .kg_subgraph_scale_eval import KGSubgraphScaleEvaluator

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/evaluate/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, cur_path, _import_structure)

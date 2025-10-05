# Lazy imports to avoid dependency issues
# Import only when needed to prevent circular dependencies and missing optional packages

__all__ = [
    'DocumentService', 'SentenceProcessingService', 'TextPreprocessingService', 'WordManagementService',
    'SentenceAnalysisService', 'WordAnalysisService', 'WordStatisticsService',
    'GraphService', 'Word2VecService', 'VisualizationService'
]

def __getattr__(name):
    """Lazy import to avoid loading all dependencies at once"""
    if name == 'DocumentService':
        from .Document.DocumentService import DocumentService
        return DocumentService
    elif name == 'SentenceProcessingService':
        from .Document.SentenceProcessingService import SentenceProcessingService
        return SentenceProcessingService
    elif name == 'TextPreprocessingService':
        from .Document.TextPreprocssingService import TextPreprocessingService
        return TextPreprocessingService
    elif name == 'WordManagementService':
        from .Document.WordManagementService import WordManagementService
        return WordManagementService
    elif name == 'SentenceAnalysisService':
        from .Sentence.SentenceAnalysisService import SentenceAnalysisService
        return SentenceAnalysisService
    elif name == 'WordAnalysisService':
        from .Word.wordAnalysisService import WordAnalysisService
        return WordAnalysisService
    elif name == 'WordStatisticsService':
        from .Word.wordStatisticsService import WordStatisticsService
        return WordStatisticsService
    elif name == 'GraphService':
        from .Graph.GraphService import GraphService
        return GraphService
    elif name == 'Word2VecService':
        from .Word2Vec.Word2VecService import Word2VecService
        return Word2VecService
    elif name == 'VisualizationService':
        from .Visualization.VisualizationService import VisualizationService
        return VisualizationService
    elif name == 'MemoryDataLoader':
        from .Word2Vec.DataLoader import MemoryDataLoader
        return MemoryDataLoader
    elif name == 'MemoryWord2vecDataset':
        from .Word2Vec.DataLoader import MemoryWord2vecDataset
        return MemoryWord2vecDataset
    elif name == 'Word2VecTrainer':
        from .Word2Vec.Trainer import Word2VecTrainer
        return Word2VecTrainer
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class DimensionReducer:
    """
    차원 축소를 담당하는 클래스.
    t-SNE, UMAP, PCA 등을 제공.
    """

    @staticmethod
    def reduce_tsne(
        embeddings: np.ndarray,
        n_components: int = 2,
        perplexity: int = 30,
        n_iter: int = 1000,
        random_state: int = 42
    ) -> np.ndarray:
        """
        t-SNE를 사용한 차원 축소.

        Args:
            embeddings: 입력 임베딩 (n_samples, n_features)
            n_components: 목표 차원 (기본 2D)
            perplexity: t-SNE perplexity 파라미터 (5~50 권장)
            n_iter: 최적화 반복 횟수
            random_state: 재현성을 위한 시드

        Returns:
            축소된 임베딩 (n_samples, n_components)
        """
        # perplexity는 샘플 수보다 작아야 함
        n_samples = embeddings.shape[0]
        perplexity = min(perplexity, n_samples - 1)

        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_state,
            init='pca',  # PCA 초기화로 안정성 향상
            learning_rate='auto'
        )

        return tsne.fit_transform(embeddings)

    @staticmethod
    def reduce_umap(
        embeddings: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42
    ) -> np.ndarray:
        """
        UMAP을 사용한 차원 축소.
        t-SNE보다 빠르고 전역 구조를 더 잘 보존.

        Args:
            embeddings: 입력 임베딩
            n_components: 목표 차원
            n_neighbors: 이웃 수 (클수록 전역 구조 강조)
            min_dist: 최소 거리 (작을수록 클러스터가 밀집)
            random_state: 재현성을 위한 시드

        Returns:
            축소된 임베딩
        """
        if not UMAP_AVAILABLE:
            raise ImportError(
                "UMAP이 설치되지 않았습니다. "
                "'pip install umap-learn'으로 설치하세요."
            )

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            metric='cosine'  # 텍스트 임베딩에 적합
        )

        return reducer.fit_transform(embeddings)

    @staticmethod
    def reduce_pca(
        embeddings: np.ndarray,
        n_components: int = 2,
        random_state: int = 42
    ) -> np.ndarray:
        """
        PCA를 사용한 선형 차원 축소.
        빠르지만 비선형 구조를 놓칠 수 있음.

        Args:
            embeddings: 입력 임베딩
            n_components: 목표 차원
            random_state: 재현성을 위한 시드

        Returns:
            축소된 임베딩
        """
        pca = PCA(n_components=n_components, random_state=random_state)
        return pca.fit_transform(embeddings)

    @staticmethod
    def auto_select_method(
        embeddings: np.ndarray,
        n_components: int = 2,
        random_state: int = 42
    ) -> tuple[np.ndarray, str]:
        """
        데이터 크기에 따라 최적의 차원 축소 방법을 자동 선택.

        Args:
            embeddings: 입력 임베딩
            n_components: 목표 차원
            random_state: 재현성

        Returns:
            (축소된 임베딩, 사용된 메서드 이름)
        """
        n_samples = embeddings.shape[0]

        if n_samples < 100:
            # 소규모: PCA (빠르고 안정적)
            reduced = DimensionReducer.reduce_pca(
                embeddings, n_components, random_state
            )
            return reduced, 'PCA'
        elif n_samples < 5000:
            # 중규모: t-SNE (디테일한 클러스터 구조)
            reduced = DimensionReducer.reduce_tsne(
                embeddings, n_components, random_state=random_state
            )
            return reduced, 't-SNE'
        else:
            # 대규모: UMAP (빠르고 확장성 좋음)
            if UMAP_AVAILABLE:
                reduced = DimensionReducer.reduce_umap(
                    embeddings, n_components, random_state=random_state
                )
                return reduced, 'UMAP'
            else:
                # UMAP 없으면 PCA로 fallback
                reduced = DimensionReducer.reduce_pca(
                    embeddings, n_components, random_state
                )
                return reduced, 'PCA (UMAP unavailable)'

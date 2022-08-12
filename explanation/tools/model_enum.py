from enum import Enum


class SurvivalModel(Enum):
    COMPWISE_GRAD_BOOST = 'ComponentwiseGradientBoostingSurvivalAnalysis'
    GRAD_BOOST = 'GradientBoostingSurvivalAnalysis'
    RSF = 'RandomSurvivalForest'
    EST = 'ExtraSurvivalTrees'
    COXNET = 'CoxnetSurvivalAnalysis'
    COXPH = 'CoxPHSurvivalAnalysis'
    AFT = 'IPCRidge'
    HINGE_SVM = 'HingeLossSurvivalSVM'
    KERNEL_SVM = 'FastKernelSurvivalSVM'
    SVM = 'FastSurvivalSVM'
    LIPSCHITZ_SVM = 'MinlipSurvivalAnalysis'
    NAIVE_SVM = 'NaiveSurvivalSVM'
    TREE = 'SurvivalTree'

    def is_ensemble(self) -> bool:
        return self in [SurvivalModel.COMPWISE_GRAD_BOOST, SurvivalModel.GRAD_BOOST,
                        SurvivalModel.RSF, SurvivalModel.EST]

    def is_linear(self) -> bool:
        return self in [SurvivalModel.COXNET, SurvivalModel.COXPH, SurvivalModel.AFT, SurvivalModel.SVM]

    def is_svm(self) -> bool:
        return self in [SurvivalModel.HINGE_SVM, SurvivalModel.KERNEL_SVM, SurvivalModel.SVM,
                        SurvivalModel.LIPSCHITZ_SVM, SurvivalModel.NAIVE_SVM]

    def can_predict_survival(self) -> bool:
        return not (self.is_svm() or self == SurvivalModel.AFT)

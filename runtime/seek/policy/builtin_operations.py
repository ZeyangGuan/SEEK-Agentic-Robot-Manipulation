from .operation_knowledge import OperationKnowledge


def initial_operation_knowledge():
    return OperationKnowledge({}, episode=0).bank

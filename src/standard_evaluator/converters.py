from standard_evaluator.problem import OptProblem
from standard_evaluator.evaluator import EvaluatorInfo

def opt_problem_to_evaluator_info(opt_prob: OptProblem) -> EvaluatorInfo:
    return EvaluatorInfo(
        name=opt_prob.name,
        inputs=opt_prob.variables,
        outputs=opt_prob.responses,
        description=opt_prob.description,
        cite=opt_prob.cite,
        options=opt_prob.options,
    )


def evaluator_info_to_opt_problem(eval_info: EvaluatorInfo) -> OptProblem:
    return OptProblem(
        name=eval_info.name,
        variables=eval_info.inputs,
        responses=eval_info.outputs,
        description=eval_info.description,
        cite=eval_info.cite,
        options=eval_info.options,
    )
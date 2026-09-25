from gymnasium.envs.registration import register

register(
    id="ims-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
)

register(
    id="ims-choice-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"choice": True},
)

register(
    id="ims-inplace-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"replacement": "inplace"},
)

register(
    id="ims-choice-inplace-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"choice": True, "replacement": "inplace"},
)

register(
    id="ims-inplace_lowest-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"replacement": "inplace_lowest"},
)

register(
    id="ims-choice-inplace_lowest-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"choice": True, "replacement": "inplace_lowest"},
)

register(
    id="ims-diff-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"objective": "diff"},
)

register(
    id="ims-choice-diff-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"choice": True, "objective": "diff"},
)

register(
    id="ims-inplace-diff-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"replacement": "inplace", "objective": "diff"},
)

register(
    id="ims-choice-inplace-diff-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"choice": True, "replacement": "inplace", "objective": "diff"},
)

register(
    id="ims-inplace_lowest-diff-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"replacement": "inplace_lowest", "objective": "diff"},
)

register(
    id="ims-choice-inplace_lowest-diff-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"choice": True, "replacement": "inplace_lowest", "objective": "diff"},
)

register(
    id="ims-var-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"objective": "var"},
)

register(
    id="ims-choice-var-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"choice": True, "objective": "var"},
)

register(
    id="ims-inplace-var-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"replacement": "inplace", "objective": "var"},
)

register(
    id="ims-choice-inplace-var-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"choice": True, "replacement": "inplace", "objective": "var"},
)

register(
    id="ims-inplace_lowest-var-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"replacement": "inplace_lowest", "objective": "var"},
)

register(
    id="ims-choice-inplace_lowest-var-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"choice": True, "replacement": "inplace_lowest", "objective": "var"},
)

register(
    id="ims-mean-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"objective": "mean_diff"},
)

register(
    id="ims-choice-mean-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"choice": True, "objective": "mean_diff"},
)

register(
    id="ims-inplace-mean-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"replacement": "inplace", "objective": "mean_diff"},
)

register(
    id="ims-choice-inplace-mean-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"choice": True, "replacement": "inplace", "objective": "mean_diff"},
)

register(
    id="ims-inplace_lowest-mean-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"replacement": "inplace_lowest", "objective": "mean_diff"},
)

register(
    id="ims-choice-inplace_lowest-mean-v0",
    entry_point="mo_gymnasium.envs.ims.ims:IteratedMeioticSelection",
    kwargs={"choice": True, "replacement": "inplace_lowest", "objective": "mean_diff"},
)
# pylint: disable=unused-import,abstract-method,unused-argument
from math import ceil

from dallinger.prolific import DevProlificService, ProlificServiceException
from dominate import tags
from psynet.end import UnsuccessfulEndLogic
from psynet.timeline import TimelineLogic
from psynet.utils import get_logger


logger = get_logger()

WAGE_PER_HOUR = 9

class CustomUnsuccessfulEndLogic(UnsuccessfulEndLogic):
    def before_debrief(self, experiment, participant):
        if experiment.with_prolific_recruitment():
            time_credit = participant.time_credit
            reward = round(WAGE_PER_HOUR*time_credit/60**2, 2)
            minimal_payment = ceil(time_credit/60) * 0.1

            if reward < minimal_payment:
                reward = minimal_payment
                logger.warning(f"Reward for participant {participant.id} is below minimum of 0.1 per minute. Setting to minimal reward.")

            study_id = participant.hit_id
            submission_id = participant.assignment_id
            params = {
                "submission_ids": [submission_id],
                "bonus_per_submission": reward,
                "increase_places": True,
            }
            screened_out = False
            if isinstance(experiment.recruiter.prolificservice, DevProlificService):
                logger.info(f"""Simulating API call to compensate failed participant {participant.id} with bonus {reward}:
                endpoint: /studies/{study_id}/screen-out-submissions/
                params: {params}
                """)
            else:
                try:
                    experiment.recruiter.prolificservice._req(method="POST", endpoint=f"/studies/{study_id}/screen-out-submissions/", json=params)
                    logger.info(f"Compensated failed participant {participant.id} with bonus {reward}.")
                    screened_out = True
                except ProlificServiceException as e:
                    logger.error(f"Failed to compensate failed participant {participant.id} with bonus {reward}.")
            participant.var.set("screened_out", screened_out)

            participant.var.set("reward", reward)
            participant.bonus = reward


    def debrief_participant(self, experiment, participant) -> TimelineLogic:
        html = tags.span()

        with html:
            tags.span("Unfortunately the experiment must end early.")

            if participant.var.get("microphone_error", False):
                tags.span("We detected that your microphone does not comply with our listed requirements for this task.")

            if experiment.with_prolific_recruitment():
                reward = participant.var.get("reward", 0)
                if participant.var.get("screened_out", False):
                    tags.span(f"You have been screened out of the study and will receive a compensation of £{reward}.")
                    tags.p("You can now close this window.")
                else:
                    tags.span("Please now return the assignment on Prolific.")
                    tags.p("After returning, you can close this window.")
                    logger.info(f"""
                    Participant {participant.id} could not be screened out.
                    assignment_id: {participant.assignment_id}
                    hit_id: {participant.hit_id}
                    reward: {reward}
                    """)

        return self.debrief_page(html, experiment, participant, show_finish_button=False)

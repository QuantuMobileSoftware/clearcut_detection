import json
from sqlalchemy import text
from db_engin.common import DataBaseResponseMixin


class RunPredictTasks(DataBaseResponseMixin):

    @classmethod
    def get_task_by_id(cls, session, task_id):
        # t = text("SELECT crut.id, crut.path_type, crut.path_img_0, crut.path_img_1, crut.image_date_0, \
        # crut.image_date_1, crut.result, crut.date_created, crut.date_started, crut.date_finished, ct.tile_index \
        # FROM clearcuts_run_update_task crut \
        #   LEFT JOIN clearcuts_tile ct on crut.tile_index_id = ct.id WHERE crut.id=:task_id").bindparams(
        #     task_id=int(task_id)
        # )
        t = text("SELECT * FROM clearcuts_run_update_task WHERE id=:task_id").bindparams(
            task_id=int(task_id)
        )

        return cls.get_row_as_dict(session, t)

    @classmethod
    def update_task_by_id(cls, session, task_id, params):
        """
        Update run_algorimt_tasks by task id, and dict values
        :param session: sqlalchemy session
        :param task_id: int
        :param params: dict (key - column name, value - value)
        :return:
        """
        t = text("SELECT *******(:task_id, :params)").bindparams(
            task_id=task_id,
            params=json.dumps(params)
        )

        q_res = session.execute(t)
        res = q_res.fetchone() if q_res.rowcount else None
        session.commit()
        return res

    @classmethod
    def add_logs(cls, session, task_id, log_level, log_message):
        t = text("SELECT * FROM ***********(:task_id, :log_level, :log_message)").bindparams(
            task_id=task_id,
            log_level=log_level,
            log_message=log_message,
        )
        return cls.get_single_obj(session, t)

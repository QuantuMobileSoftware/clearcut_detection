from sqlalchemy import text
from db_engin.common import DataBaseResponseMixin


class RunPredictTasks(DataBaseResponseMixin):

    @classmethod
    def get_task_by_id(cls, session, task_id):
        t = text("SELECT * FROM clearcuts_run_update_task WHERE id=:task_id").bindparams(
            task_id=int(task_id)
        )

        return cls.get_row_as_dict(session, t)

    @classmethod
    def update_task_by_id(cls, session, task_id, params):
        """
        Update result field of clearcuts_run_update_task table by task id
        :param session: sqlalchemy session
        :param task_id: int
        :param params: dict
        :return:
        """
        print(f'params: {params}')

        t = text("UPDATE clearcuts_run_update_task SET result=:result, date_started=:date_started, date_finished=:date_finished \
        WHERE id=:task_id").bindparams(
            task_id=task_id,
            result=params['result'],
            date_started=params['date_started'],
            date_finished=params['date_finished'],
        )
        session.execute(t)
        session.commit()

    @classmethod
    def update_tileinformation(cls, session, tile_index_id):
        """
        Update clearcuts_tileinformation table by tile_index
        set is_predicted=1.
        :param session: sqlalchemy session
        :param tile_index_id: int
        :return:
        """
        t = text('UPDATE clearcuts_tileinformation SET is_predicted=1 WHERE tile_index_id=:tile_index_id').bindparams(
            tile_index_id=tile_index_id
        )
        session.execute(t)
        session.commit()

    @classmethod
    def add_logs(cls, session, task_id, log_level, log_message):
        t = text("SELECT * FROM ***********(:task_id, :log_level, :log_message)").bindparams(
            task_id=task_id,
            log_level=log_level,
            log_message=log_message,
        )
        return cls.get_single_obj(session, t)

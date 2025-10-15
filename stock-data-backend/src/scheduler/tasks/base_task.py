from abc import ABC, abstractmethod
from apscheduler.schedulers.background import BackgroundScheduler
from data.storage import Storage

class ScheduledTask(ABC):
    def __init__(self, scheduler: BackgroundScheduler, storage: Storage):
        self.scheduler = scheduler
        self.storage = storage

    @abstractmethod
    def run(self):
        pass
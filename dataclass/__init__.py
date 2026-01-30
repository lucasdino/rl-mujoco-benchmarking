from dataclass.replay_buffer import ReplayBuffer
from dataclass.priority_experience_replay import PriorityExperienceReplay


BUFFER_MAPPING = {
	"replay_buffer": ReplayBuffer,
	"priority_experience_replay": PriorityExperienceReplay,
}

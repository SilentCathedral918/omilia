import os
import pickle
import gzip

class Mnimi:
	def __init__(self, short_term_max_length=10, data_pkl='mnimi.pkl', compressed=True):
		self._short_term = []
		self._short_term_max_length = short_term_max_length
		self._data_pkl = data_pkl
		self._compressed = compressed

		self.load_mnimi()

	def load_mnimi(self):
		path_ = self._data_pkl

		if not path_.endswith('.pkl'):
			path_ = f'{path_}.pkl'
			
		if not self._compressed and not os.path.exists(path_):
			self._long_term = {}
			self._knowledge = {}
			return
		
		if self._compressed and not os.path.exists(f'{path_}.gz'):
			self._long_term = {}
			self._knowledge = {}
			return

		file_path_ = f'{path_}.gz' if self._compressed else path_
		
		with gzip.open(file_path_, 'rb') if self._compressed else open(file_path_, 'rb') as file_:
			mnimi_data_ = pickle.load(file_)
			self._long_term = mnimi_data_.get('long_term', {})
			self._knowledge = mnimi_data_.get('knowledge', {})

	def save_mnimi(self):
		path_ = self._data_pkl

		if not path_.endswith('.pkl'):
			path_ = f'{path_}.pkl'
		
		file_path_ = f'{path_}.gz' if self._compressed else path_

		mnimi_data_ = {
			'long_term': self._long_term,
			'knowledge': self._knowledge
		}

		with gzip.open(file_path_, 'wb') if self._compressed else open(file_path_, 'wb') as file_:
			pickle.dump(mnimi_data_, file_, protocol= 5 if self._compressed else pickle.DEFAULT_PROTOCOL)

	def store_short_term(self, input, response):
		self._short_term.append((input, response))
		if len(self._short_term) > self._short_term_max_length:
			self._short_term.pop(0)

	def store_long_term(self, key, value, override=False):
		if not override and key in self._long_term:
			print(f"Long term memory of key '{key}' already exists. Set override=True to overwrite data.")
		else:	
			self._long_term[key] = value
			self.save_mnimi()

	def store_knowledge(self, key, value, override=False):
		if not override and key in self._knowledge:
			print(f"Knowledge of key '{key}' already exists. Set override=True to overwrite data.")
		else:	
			self._knowledge[key] = value
			self.save_mnimi()

	def remove_long_term(self, key):
		self._long_term.pop(key)
		self.save_mnimi()

	def remove_knowledge(self, key):
		self._knowledge.pop(key)
		self.save_mnimi()

	def clear_short_term(self):
		self._short_term = []

	def clear_long_term(self):
		self._long_term.clear()
		self.save_mnimi()
	
	def clear_knowledge(self):
		self._knowledge.clear()
		self.save_mnimi()

	def get_short_term(self):
		return self._short_term
	
	def get_long_term(self, key):
		if key in self._long_term:
			return self._long_term.get(key, None)
		else:
			print(f"Long term memory of key '{key}' is not found.")
			return None

	def get_knowledge(self, key):
		if key in self._knowledge:
			return self._knowledge.get(key, None)
		else:
			print(f"Knowledge of key '{key}' is not found.")
			return None
		
	def get_long_term_keys(self):
		return list(self._long_term.keys())

	def get_knowledge_keys(self):
		return list(self._knowledge.keys())

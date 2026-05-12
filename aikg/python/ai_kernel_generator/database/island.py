import os
import json
import logging
import shutil
import random
from pathlib import Path

from ai_kernel_generator.utils.common_utils import get_md5_hash

from ai_kernel_generator.database.database import Database
from ai_kernel_generator.database.program import Program
from ai_kernel_generator.database.evolve_database import EvolveVectorStore

logger = logging.getLogger(__name__)

class Island(Database):
    def __init__(self, island_id: int, pd_path: str, database_config: dict, evolve_shortcut: list[str], evolve_database: str):
        self.evolve_database_suffix = evolve_database
        self.island_database_path = pd_path + '/island_' + str(island_id)
        self.island_id = island_id
        os.makedirs(self.island_database_path, exist_ok=True)
        
        # maintain an online list
        self.program_list: list[Program] = []
        
        # maintain an program list of current task round
        self.program_list_current_round: list[Program] = []
        
        self.move_evolve_shortcut(evolve_shortcut)
        
        self.basic_vector_store = EvolveVectorStore(
            database_path=self.island_database_path,
            embedding_model_name='/mnt/lustre-client/zhangzizheng/ALL_MODELS/Jerry0/text2vec-large-chinese',
            index_name='basic_vector_store',
            features=['basic'],
            config=database_config
        )
        self.schedule_vector_store = EvolveVectorStore(
            database_path=self.island_database_path,
            embedding_model_name='/mnt/lustre-client/zhangzizheng/ALL_MODELS/Jerry0/text2vec-large-chinese',
            index_name='schedule_vector_store',
            features=['schedule'],
            config=database_config
        )
        self.memory_vector_store = EvolveVectorStore(
            database_path=self.island_database_path,
            embedding_model_name='/mnt/lustre-client/zhangzizheng/ALL_MODELS/Jerry0/text2vec-large-chinese',
            index_name='memory_vector_store',
            features=['memory'],
            config=database_config
        )
        self.vector_stores = [self.basic_vector_store, self.schedule_vector_store, self.memory_vector_store]
        self.vector_store_map = {
            self.basic_vector_store: 'basic',
            self.schedule_vector_store: 'schedule',
            self.memory_vector_store: 'memory',
        }
                
        super().__init__(self.island_database_path, self.vector_stores, database_config)
        
        logger.info(f'Island {island_id} was created, has {len(evolve_shortcut)} evolve shortcuts.\n')
    
    def move_evolve_shortcut(self, evolve_shortcut: list[str]):
        if len(evolve_shortcut) == 0:
            return
        checkpoint_parent_id = open(Path(self.get_checkpoint_path()) / "checkpoint.txt", "r").readline().strip()
        for es in evolve_shortcut:
            src_dir = Path(self.get_checkpoint_path()) / es
            if os.path.exists(src_dir) and os.path.isdir(src_dir):
                des_dir = Path(self.island_database_path) / es
                # 剔除检查点parent_id下面的算子（在上一次检查点后没有被完整记录）
                if Program(str(src_dir)).get_parent_id() == checkpoint_parent_id:
                    continue
                os.system(f"cp -r {src_dir} {des_dir}")      
                # maintain an online list
                self.program_list.append(Program(str(des_dir)))    
    
    def sample_latest(self):
        if len(self.program_list) == 0:
            return None
        return self.program_list[-1].get_impl_info()
    
    def random_sample_parent(self):
        if len(self.program_list) == 0:
            return None
        self.program_list.sort(key=lambda x: int(x.get_impl_info()["profile"]["gen_time"]))
        return random.choice(self.program_list).get_impl_info()
    
    def random_sample_others(self, parent_id: int, sample_num: int):
        if len(self.program_list) <= 1:
            return []
        if len(self.program_list) < 1 + sample_num:
            sample_num = len(self.program_list) - 1
        program_exclude_parent_list = [p for p in self.program_list if p.get_impl_info()['id'] != parent_id]
        return [ p.get_impl_info() for p in random.choices(program_exclude_parent_list, k=sample_num)]

    def get_exist_code_ir(self) -> list[str]:
        exist_code_ir = []
        for program in self.program_list:
            impl_info = program.get_impl_info()
            exist_code_ir.append(impl_info['sketch'])
        return exist_code_ir
    
    def get_exist_code_feat(self) -> list[str]:
        exist_code_feat = []
        for program in self.program_list:
            code_feat = program.get_impl_feat()
            exist_code_feat.append(json.dumps(code_feat, ensure_ascii=False))
        return exist_code_feat
    
    async def insert(self, impl_code: str, framework_code: str, profile: str, backend: str, arch: str, dsl: str, impl_info: dict):
        
        # insert into FAISS
        md5_hash = get_md5_hash(impl_code=impl_code)
        file_path = Path(self.island_database_path) / md5_hash
        
        if file_path.exists():
            self.program_list.remove(Program(file_path))

        import os
        if os.environ.get('AIKG_DEBUG_MODE', False):
            features = json.load(open('/mnt/lustre-client/zhangzizheng/AIKG/akg/aikg/examples/debug_io/example_output/20c850f9/island_0/metadata.json', 'r'))
        else:
            features = await self.extract_features('', impl_code, framework_code, backend, arch, dsl, '', profile)
            
        file_path.mkdir(parents=True, exist_ok=True)
        metadata_file = file_path / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False, indent=4)

        impl_file = file_path / "impl_code.py"
        with open(impl_file, "w", encoding="utf-8") as f:
            f.write(impl_code)

        for vector_store in self.vector_stores:
            vector_store.insert(f"{md5_hash}")
        
        
        info_data_path = file_path / "impl_info.json"
        with open(info_data_path, 'w', encoding='utf-8') as f:
            json.dump(impl_info, f, ensure_ascii=False, indent=2)
        
        # maintain an online list
        self.program_list.append(Program(file_path))
        
        # add program to offline evolve database (checkpoint)
        src_dir = file_path
        des_dir = Path(self.get_checkpoint_path()) / os.path.basename(src_dir)
        shutil.copytree(src_dir, des_dir, dirs_exist_ok=True) 
            
        
        logger.info(f"Operator implementation inserted successfully, file path: {file_path}")
    
    def update_early_stopping_reason(self, program_id: str, reason: str):
        program = self.find_program_by_id(program_id)
        if program:
            impl_info = program.get_impl_info()
            impl_info["early_stopping_reason"] = reason
            with open(Path(program.file_dir) / "impl_info.json", 'w', encoding='utf-8') as f:
                json.dump(impl_info, f, ensure_ascii=False, indent=2)

    def find_program_by_id(self, id: str) -> Program:
        for p in self.program_list:
            if id == p.get_impl_info()['id']:
                return p
        return None
    
    def get_branch_from_root(self, target_program_id: str) -> list[Program]:
        """
        根据指定program_id，返回从根节点到该节点的算子树分支列表（根→子→目标节点顺序）
        
        参数:
            target_program_id: 目标算子的唯一id
        返回:
            List[Program]: 从根到目标节点的Program列表，顺序为根→父→子→目标节点；若节点不存在返回空列表
        """
        # 1. 初始化结果列表，先找到目标节点
        branch = []
        current_program = self.find_program_by_id(target_program_id)
        
        # 目标节点不存在，直接返回空列表
        if not current_program:
            return branch
        
        # 2. 反向溯源：从目标节点找父代，直到根节点（parent_id为空/None）
        while current_program:
            branch.append(current_program)
            parent_id = current_program.get_impl_info()["parent_id"]
            
            # 父代id为空，说明当前是根节点，终止循环
            if not parent_id:
                break
            
            # 查找父代节点
            current_program = self.find_program_by_id(parent_id)
            
            # 防呆：如果父代id存在但找不到对应节点（数据异常），终止循环
            if not current_program:
                break
        
        # 3. 反转列表，得到从根到目标节点的顺序
        branch.reverse()
        
        return branch

    def get_parent_id(self, program_id: str) -> str:
        program = self.find_program_by_id(program_id)
        for p in self.program_list:
            if p.get_impl_info().get("id") == program.get_impl_info().get("parent_id"):
                return p.get_impl_info().get("id")
        return None
    
    def get_child_list(self, program_id: str) -> list[str]:
        # 返回该算子的孩子节点列表，以speedup降序排序
        program = self.find_program_by_id(program_id)
        child_list = []
        for p in self.program_list:
            if p.get_impl_info().get("parent_id") == program.get_impl_info().get("id"):
                child_list.append(p)
        sorted_list = sorted(
            child_list,
            key=lambda p: p.get_impl_info().get("profile", {}).get("speedup", 0),  # 逐层get避免KeyError
            reverse=True  # 降序排序
        )
        return [p.get_impl_info().get("id") for p in sorted_list]
    
    def has_child(self, program_id:str) -> bool:
        program = self.find_program_by_id(program_id)
        for p in self.program_list:
            if p.get_impl_info().get("parent_id") == program.get_impl_info().get("id"):
                return True
        return False
    
    def get_fallback_candidate_list(self, stop_program_id: str, fallback_depth: int) -> list[str]:
        parent_id = self.get_parent_id(stop_program_id)
        while parent_id is not None and fallback_depth > 0:
            parent_id = self.get_parent_id(stop_program_id)
            fallback_depth -= 1
        
        if parent_id is not None:
            child_list = self.get_child_list(parent_id)
            # 去除有孩子节点的child node，因为其之前被探索过了已经（代表曾经回退过）
            return [c for c in child_list if self.has_child(c) is False]
            
    def get_inefficiency_programs(self) -> list[dict]:
        # 低效算子是指 profile 中 speedup <= 1.1x 的算子
        return [p.get_impl_info() for p in self.program_list if p.get_impl_info().get("profile", {}).get("speedup", 0) <= 1.1]

    def get_checkpoint_path(self) -> str:
        return str(
            Path(self.island_database_path).parent.parent.parent / "evolve_database" / self.evolve_database_suffix / f"island_{self.island_id}"
        )
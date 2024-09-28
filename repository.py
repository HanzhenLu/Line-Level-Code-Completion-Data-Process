import ast
import json
import os
import pandas as pd
from typing import Dict, List, Set, Tuple
from collections import deque
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

output_path = "/data/hanzhenlu/dataset"
max_workers = 40

class File:
    def __init__(self, path:str, content:str) -> None:
        self.path = path
        self.content = content
        self.dep_in = 0
        self.dep_out_path: set = set()
        self.dep_in_lib: set = set()
        self.dep_in_path: set = set()
        self.subgraph_idx: int = -1
    
    # 方便打印出来查看内容
    def __str__(self) -> str:
        return "path:{}\ndep_in:{}\ndep_out_path:{}".format(self.path, self.dep_in, str(self.dep_out_path))
    
    # 用于set
    def __hash__(self) -> int:
        return str(self).__hash__()

# 提取代码中的依赖关系
def extract_imports(content: str) -> List:
    # 为了防止<UNK>导致解析失败，使用zxcv替换
    try:
        code = content.replace("<UNK>", "zxcv")
        tree = ast.parse(code)
    except Exception as e:
        print(code)
        print(e)
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    
    return imports

def TopologicalSort(project:Dict[str, str]) -> List[List[File]]:
    graph: Dict[str, File] = {}
    for path, content in project.items():
        cur_file = File(path, content)
        imports = extract_imports(content)
        for imp in imports:
            cur_file.dep_in_lib.add(imp)
        graph[path] = cur_file
    
    # 根据依赖关系连接图中的各个节点
    for key, value in graph.items():
        for imp in value.dep_in_lib:
            imp = imp.replace('.', '/')
            for file in graph.keys():
                if file.endswith(imp+"/__init__.py") or file.endswith(imp+".py"):
                    graph[file].dep_out_path.add(key)
                    graph[key].dep_in_path.add(file)
                    graph[key].dep_in += 1

    # 将整个图分为互不连通的子图
    subgraphs: List[Set[File]] = []
    for _, node in graph.items():
        if node.subgraph_idx == -1:
            subgraph = set()
            subgraph.add(node)
            idx = len(subgraphs)
            node.subgraph_idx = idx
            queue = deque()
            
            for file_path in node.dep_out_path:
                queue.append(file_path)
            for file_path in node.dep_in_path:
                queue.append(file_path)
            while queue:
                element = queue.popleft()
                element = graph[element]
                if element.subgraph_idx == -1:
                    subgraph.add(element)
                    element.subgraph_idx = idx
                    for file_path in node.dep_out_path:
                        queue.append(file_path)
                    for file_path in node.dep_in_path:
                        queue.append(file_path)
            subgraphs.append(subgraph)
            

    def find_min_in_node(subgraph: Set[File], results: List[File]) -> File:
        min_in = 0
        min_node = None
        for node in subgraph:
            if node not in results:
                if min_node is None or node.dep_in < min_in:
                    min_node = node
                    min_in = node.dep_in
        
        return min_node
    
    # 将每个子图转化为一个字符串
    all_results = []
    for subgraph in subgraphs:
        results = []
        while len(results) != len(subgraph):
            min_node = find_min_in_node(subgraph, results)
            for out_file in min_node.dep_out_path:
                graph[out_file].dep_in -= 1
            results.append(min_node)
        
        all_results.append(results)
        
    return all_results

def process_project(project_tuple:List[Tuple[str, str]]):
    project_dict = {}
    for tuple in project_tuple:
        project_dict[tuple[0]] = tuple[1]
    res = TopologicalSort(project_dict)
    sample = ""
    for subgraph in res:
        for file in subgraph:
            # 附加路径信息
            sample += "# " + file.path + '\n'
            sample += file.content + '\n'
    return {"text":sample}

if __name__ == "__main__":
    with open("repo.json", 'r') as f:
        js = json.loads(f.read())

    samples = []
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     futures = [executor.submit(process_project, project_tuple) for _, project_tuple in tqdm(js.items())]
    
    # for future in as_completed(futures):
    #     samples.append(future.result())
    
    for _, project_tuple in tqdm(js.items()):
        samples.append(process_project(project_tuple))
             
    output_file_path = os.path.join(output_path, "Stack-V2-python-repository.parquet")
    df = pd.DataFrame(samples)
    print("writing to the file")
    df.to_parquet(output_file_path)

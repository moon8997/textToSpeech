import pymysql
from dataclasses import dataclass

from .util import load_config

@dataclass
class Gate:
    gate_name: str
    gate_seq: int

@dataclass
class PushBack:
    pushback_seq: int
    gate: int
    pushback: str
    text: str

    
def fetch_data():
    # Read database configuration from config.ini
    config = load_config('config.ini')

    db_config = config['DBConfig']

    # Connect to the database using DictCursor
    connection = pymysql.connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['passwd'],
        db=db_config['db'],
        charset=db_config['charset'],
        port=int(db_config['port']),
        cursorclass=pymysql.cursors.DictCursor
    )

    gates = []
    pushBacks = []

    try:
        with connection.cursor() as cursor:
            # 쿼리 실행
            query = """
            SELECT as_name as gateName, as_gate_cnt as gateSeq
            FROM ast_gate
            WHERE as_airport_seq = 1
            ORDER BY CAST(as_name AS SIGNED)
            """
            cursor.execute(query)
            
            # 결과 가져오기
            results = cursor.fetchall()
            
            # Gate 객체로 결과 저장
            for row in results:
                gate = Gate(row['gateName'], row['gateSeq'])
                gates.append(gate)

            query = """
            select pushback_seq, gate, pushback, text
            from ast_pushback
            """
            cursor.execute(query)
            
            # 결과 가져오기
            results = cursor.fetchall()
            
            # PushBack 객체로 결과 저장
            for row in results:
                pushBack = PushBack(row['pushback_seq'], int(row['gate']), row['pushback'], row['text'])
                pushBacks.append(pushBack)
    finally:
        connection.close()
    
    return gates, pushBacks
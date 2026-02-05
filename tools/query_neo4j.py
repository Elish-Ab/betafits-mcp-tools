from neo4j import GraphDatabase
import os

class Neo4jClient:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_latest_entity(self):
        query = """
        MATCH (a)-[r]->(b)
        WHERE a.created_at IS NOT NULL
        RETURN b
        ORDER BY a.created_at DESC
        LIMIT 1;
        """

        with self.driver.session() as session:
            return session.run(query).single()

if __name__ == "__main__":
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    neo = Neo4jClient(uri, user, password)
    data=neo.get_latest_entity()
    print(data)
    print("*" * 20)
    print("Latest Entity:", data.get("b")["uuid"] if data else "No entity found")
    neo.close()

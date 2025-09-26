import json
from typing import Dict, Any
from kafka import KafkaProducer, KafkaConsumer
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)

class StreamingHandler:
    def __init__(self, config: Dict):
        self.config = config
        self.brokers = config['storage'].get('kafka_brokers', ['localhost:9092'])
        
        self.producer = None
        self.consumers = {}
        
        self._init_producer()
    
    def _init_producer(self):
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.brokers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                compression_type='gzip',
                batch_size=16384,
                linger_ms=10
            )
            logger.info("Kafka producer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
    
    async def publish(self, topic: str, message: Dict, key: str = None):
        if not self.producer:
            return
        
        try:
            message['timestamp'] = datetime.now().isoformat()
            
            future = self.producer.send(
                topic,
                value=message,
                key=key
            )
            
            self.producer.flush()
            
            logger.debug(f"Published to {topic}: {message.get('type', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
    
    async def publish_batch(self, topic: str, messages: List[Dict]):
        if not self.producer:
            return
        
        for message in messages:
            message['timestamp'] = datetime.now().isoformat()
            
            try:
                self.producer.send(
                    topic,
                    value=message,
                    key=message.get('hostname')
                )
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
        
        self.producer.flush()
        logger.info(f"Published batch of {len(messages)} messages to {topic}")
    
    def subscribe(self, topic: str, group_id: str = 'cmdb_consumer'):
        if topic in self.consumers:
            return self.consumers[topic]
        
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.brokers,
                group_id=group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            
            self.consumers[topic] = consumer
            logger.info(f"Subscribed to topic: {topic}")
            
            return consumer
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {topic}: {e}")
            return None
    
    async def consume_messages(self, topic: str, handler_func, max_messages: int = None):
        consumer = self.subscribe(topic)
        
        if not consumer:
            return
        
        message_count = 0
        
        try:
            for message in consumer:
                await handler_func(message.value)
                
                message_count += 1
                
                if max_messages and message_count >= max_messages:
                    break
                    
        except Exception as e:
            logger.error(f"Error consuming messages: {e}")
        finally:
            consumer.close()
    
    async def stream_discoveries(self, discovered_hosts: Dict):
        topic = 'cmdb_discoveries'
        
        for hostname, data in discovered_hosts.items():
            message = {
                'event_type': 'host_discovered',
                'hostname': hostname,
                'confidence': data.get('confidence', 0),
                'occurrences': len(data.get('occurrences', [])),
                'attributes_count': len(data.get('attributes', {}))
            }
            
            await self.publish(topic, message, key=hostname)
    
    async def stream_classifications(self, classifications: Dict):
        topic = 'cmdb_classifications'
        
        for column_name, classification in classifications.items():
            message = {
                'event_type': 'column_classified',
                'column': column_name,
                'type': classification.get('type'),
                'confidence': classification.get('confidence'),
                'model': classification.get('model')
            }
            
            await self.publish(topic, message, key=column_name)
    
    async def stream_relationships(self, relationships: List[Dict]):
        topic = 'cmdb_relationships'
        
        for rel in relationships:
            message = {
                'event_type': 'relationship_discovered',
                'source': rel['source'],
                'target': rel['target'],
                'type': rel['type'],
                'confidence': rel.get('confidence', 0.5)
            }
            
            await self.publish(topic, message)
    
    def get_metrics(self) -> Dict:
        metrics = {}
        
        if self.producer:
            metrics['producer'] = self.producer.metrics()
        
        for topic, consumer in self.consumers.items():
            metrics[f'consumer_{topic}'] = consumer.metrics()
        
        return metrics
    
    def close(self):
        if self.producer:
            self.producer.close()
            logger.info("Kafka producer closed")
        
        for consumer in self.consumers.values():
            consumer.close()
        
        logger.info("Streaming handler closed")
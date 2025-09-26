# storage/streaming_handler.py
"""
Streaming Handler - Handles Kafka streaming with proper error handling
"""

import json
from typing import Dict, Any, List
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)

class StreamingHandler:
    def __init__(self, config: Dict):
        self.config = config
        
        # Safely get storage config
        storage_config = config.get('storage', {})
        self.brokers = storage_config.get('kafka_brokers', [])
        
        self.producer = None
        self.consumers = {}
        
        # Only initialize if brokers are properly configured
        if self.brokers and len(self.brokers) > 0 and self.brokers[0] and self.brokers[0] != "":
            self._init_producer()
        else:
            logger.info("Kafka streaming disabled - no brokers configured")
    
    def _init_producer(self):
        """Initialize Kafka producer if available"""
        try:
            from kafka import KafkaProducer
            
            self.producer = KafkaProducer(
                bootstrap_servers=self.brokers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                compression_type='gzip',
                batch_size=16384,
                linger_ms=10,
                max_block_ms=1000  # Don't wait forever if Kafka is down
            )
            logger.info(f"Kafka producer initialized with brokers: {self.brokers}")
        except ImportError:
            logger.info("kafka-python not installed. Install with: pip install kafka-python")
            self.producer = None
        except Exception as e:
            logger.warning(f"Kafka not available: {e}. Continuing without streaming.")
            self.producer = None
    
    async def publish(self, topic: str, message: Dict, key: str = None):
        """Publish message to Kafka topic"""
        if not self.producer:
            return
        
        try:
            message['timestamp'] = datetime.now().isoformat()
            
            future = self.producer.send(
                topic,
                value=message,
                key=key
            )
            
            self.producer.flush(timeout=1)  # Don't wait forever
            
            logger.debug(f"Published to {topic}: {message.get('type', 'unknown')}")
            
        except Exception as e:
            logger.debug(f"Could not publish message: {e}")
    
    async def publish_batch(self, topic: str, messages: List[Dict]):
        """Publish batch of messages"""
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
                logger.debug(f"Could not send message: {e}")
        
        try:
            self.producer.flush(timeout=1)
            logger.info(f"Published batch of {len(messages)} messages to {topic}")
        except:
            pass
    
    def subscribe(self, topic: str, group_id: str = 'cmdb_consumer'):
        """Subscribe to Kafka topic"""
        if not self.brokers or len(self.brokers) == 0 or not self.brokers[0]:
            return None
            
        if topic in self.consumers:
            return self.consumers[topic]
        
        try:
            from kafka import KafkaConsumer
            
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.brokers,
                group_id=group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                consumer_timeout_ms=1000  # Don't wait forever
            )
            
            self.consumers[topic] = consumer
            logger.info(f"Subscribed to topic: {topic}")
            
            return consumer
            
        except ImportError:
            logger.debug("kafka-python not installed")
            return None
        except Exception as e:
            logger.debug(f"Could not subscribe to {topic}: {e}")
            return None
    
    async def consume_messages(self, topic: str, handler_func, max_messages: int = None):
        """Consume messages from Kafka topic"""
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
            logger.debug(f"Error consuming messages: {e}")
        finally:
            consumer.close()
    
    async def stream_discoveries(self, discovered_hosts: Dict):
        """Stream discovered hosts"""
        if not self.producer:
            return
            
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
        """Stream column classifications"""
        if not self.producer:
            return
            
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
        """Stream discovered relationships"""
        if not self.producer:
            return
            
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
        """Get Kafka metrics"""
        metrics = {}
        
        if self.producer:
            try:
                metrics['producer'] = self.producer.metrics()
            except:
                pass
        
        for topic, consumer in self.consumers.items():
            try:
                metrics[f'consumer_{topic}'] = consumer.metrics()
            except:
                pass
        
        return metrics
    
    def close(self):
        """Close all Kafka connections"""
        if self.producer:
            try:
                self.producer.close()
                logger.info("Kafka producer closed")
            except:
                pass
        
        for consumer in self.consumers.values():
            try:
                consumer.close()
            except:
                pass
        
        logger.info("Streaming handler closed")
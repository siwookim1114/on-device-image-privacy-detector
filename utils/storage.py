"""
Storage Layer for Privacy Guard integrated with MongoDB
Handles encrypted face databse, privacy profiles, and provenance logs
"""
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from cryptography.fernet import Fernet
import numpy as np
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError

# Utils import
from utils.models import (
    PrivacyProfile, PersonEntry, FaceEmbedding, ConsentHistory, ProvenanceLog
)

# Encryption Manager
class EncryptionManager:
    """Manages encryption for sensitive data"""
    def __init__(self, key_path: str):
        """
        Initialize encryption manager that will handle sensitive data

        Args:
            key_path: Path to encryption key file
        """
        self.key_path = Path(key_path)
        self.key = self.load_or_create_key()
        self.cipher = Fernet(self.key)
    
    def load_or_create_key(self):
        """
        Load existing encryption key or create new one

        Returns:
            Encryption key bytes
        """
        if self.key_path.exists():
            # Load existing key
            with open(self.key_path, "rb") as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            # Save key to file
            self.key_path.parent.mkdir(parents = True, exist_ok = True)
            with open(self.key_path, "wb") as f:
                f.write(key)
            print(f"Generated new encryption key at {self.key_path}")
            return key

    def encrypt(self, data: bytes):
        """
        Encrypt data

        Args:
            data: Raw bytes to encrypt
        
        Returns:
            Encrypted bytes
        """
        return self.cipher.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes):
        """
        Decrypt data

        Args:
            encrypted_data: Encrypted bytes

        Returns:
            Decrypted bytes
        """
        return self.cipher.decrypt(encrypted_data)

# Face Database (MongoDB)
class FaceDatabase:
    """
    Face database for storing face embeddings and consent history
    Uses MongoDB with optional encryption for embeddings
    """
    def __init__(
            self, 
            mongo_uri: str = "mongodb://localhost:27017/", 
            database_name: str = "privacy_guard",
            encryption_key_path: str = "data/face_db/.encryption_key",
            encryption_enabled: bool = True
        ):
        """
        Initialize face database

        Args:
            mongo_uri: MongoDB connection URI
            database_name: Name of the database
            encryption_key_path: Path to encryption key
            encryption_enabled: Whether to encrypt embeddings
        """
        self.mongo_uri = mongo_uri
        self.database_name = database_name
        self.encryption_enabled = encryption_enabled

        # Setup encryption
        if encryption_enabled:
            self.encryptor = EncryptionManager(encryption_key_path)
        else:
            self.encryptor = None
        
        # Connect to MongoDB
        self.connect()
        print(f"Face database initialized (MongoDB: {database_name})")
    
    def connect(self):
        """Connect to MongoDB and setup collections"""
        try:
            # Create MongoDB client
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS = 5000)
            # Test Connection
            self.client.admin.command("ping")
            # Get Database and collections
            self.db = self.client[self.database_name]
            self.persons = self.db["persons"]
            self.embeddings = self.db["embeddings"]
            self.consent_history = self.db["consent_history"]

            # Create indexes
            self.create_indexes()
            print(f"Connected to MongoDB at {self.mongo_uri}")

        except ConnectionFailure as e:
            print(f"Failed to connect to MongoDB: {e}")
            raise
    
    def create_indexes(self):
        """Create indexes for faster queries"""
        # Index on person_id and label 
        self.persons.create_index([("person_id", ASCENDING)], unique = True)
        self.persons.create_index([("label", ASCENDING)])
        # Index on embeddings by person_id
        self.embeddings.create_index([("person_id", ASCENDING)])
        # Index on consent history 
        self.consent_history.create_index([("person_id", ASCENDING)], unique = True)

    def serialize_embedding(self, embedding: List[float]):
        """
        Serialize and optionally encrypt embedding

        Args:
            embedding: List of 512 floats
        Returns:
            Binary data (encrypted or plain)
        """
        arr = np.array(embedding, dtype=np.float32)
        serialized = arr.tobytes()

        # Encrypt 
        if self.encryption_enabled:
            return self.encryptor.encrypt(serialized)
        return serialized

    def deserialize_embedding(self, data: bytes) -> List:
        """
        Deserialize and optionally decrypt embedding

        Args:
            data: Encrypted or plain bytes
        Returns:
            List of 512 floats
        """
        # Decrypt if enabled
        if self.encryption_enabled:
            data = self.encryptor.decrypt(data)
        
        # Deserialized from bytes
        arr = np.frombuffer(data, dtype=np.float32)
        return arr.tolist()
    
    def add_person(self, person: PersonEntry) -> bool:
        """
        Add a new person to the database

        Args:
            person: PersonEntry object

        Returns:
            True if successful
        """
        try:
            # Prepare person document
            person_doc = {
                "person_id": person.person_id,
                "label": person.label,
                "relationship": person.relationship,
                "risk_decay_factor": person.risk_decay_factor,
                "first_seen": person.first_seen,
                "last_seen": person.last_seen,
                "notes": person.notes,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            # Insert person and embeddings
            self.persons.insert_one(person_doc)
            for emb in person.embeddings:
                embedding_doc = {
                    "person_id": person.person_id,
                    "embedding": self.serialize_embedding(emb.embedding),
                    "source_image": emb.source_image,
                    "timestamp": emb.timestamp
                }
                self.embeddings.insert_one(embedding_doc)
            # Insert consent history
            consent_doc = {
                "person_id": person.person_id,
                "times_appeared": person.consent_history.times_appeared,
                "times_approved": person.consent_history.times_approved,
                "times_protected": person.consent_history.times_protected,
                "contexts": person.consent_history.context,
                "last_consent_decision": person.consent_history.last_consent_decision,
                "consent_confidence": person.consent_history.consent_confidence
            }
            self.consent_history.insert_one(consent_doc)
            print(f"Added person: {person.label} ({person.person_id})")
            return True
        
        except DuplicateKeyError:
            print(f"Person with ID {person.person_id} already exists")
            return False
        except Exception as e:
            print(f"Failed to add person: {e}")
            return False
    
    def get_person(self, person_id: str) -> PersonEntry:
        """
        Get person by ID

        Args:
            person_id: Person's unique ID
        
        Returns:
            PersonEntry object or None
        """
        try:
            # Get person document
            person_doc = self.persons.find_one({"person_id": person_id})
            if not person_doc:
                return None
            
            # Get embeddings
            embedding_docs = self.embeddings.find({"person_id": person_id})
            embeddings = []
            for doc in embedding_docs:
                embeddings.append(FaceEmbedding(
                    embedding = self.deserialize_embedding(doc["embedding"]),
                    source_image = doc.get("source_image"),
                    timestamp = doc["timestamp"]
                ))
            # Get consent history
            consent_doc = self.consent_history.find_one({"person_id": person_id})
            consent_history = ConsentHistory(
                times_appeared = consent_doc["times_appeared"],
                times_approved = consent_doc["times_approved"],
                times_protected = consent_doc["times_protected"],
                contexts = consent_doc.get("contexts", []),
                last_consent_decision = consent_doc.get("last_consent_decision"),
                consent_confidence = consent_doc.get("consent_confidence, 0.0")
            )

            # Construct PersonEntry
            person = PersonEntry(
                person_id = person_doc["person_id"],
                label = person_doc["label"],
                relationship = person_doc["relationship"],
                embeddings = embeddings,
                consent_history = consent_history,
                risk_decay_factor = person_doc.get("risk_decay_factor", 1.0),
                first_seen = person_doc["first_seen"],
                last_seen = person_doc["last_seen"],
                notes = person_doc.get("notes")
            )
            return person
        except Exception as e: 
            print(f"Failed to get person {person_id}: {e}")
            return None
    
    def get_all_persons(self) -> Optional[List[PersonEntry]]:
        """
        Get all persons in database
        
        Returns:
            List of PersonEntry objects
        """
        try:
            person_docs = self.persons.find()
            person_ids = [doc["person_id"] for doc in person_docs]
            persons = []

            for id in person_ids:
                person = self.get_person(id)
                if person:
                    persons.append(person)
            return persons

        except Exception as e:
            print(f"Failed to get all persons: {e}")
            return []
    
    def update_person(self, person: PersonEntry) -> bool:
        """
        Update person data

        Args:
            person: Updated PersonEntry object
        
        Returns:
            True if successful
        """
        try:
            # Update person document
            self.persons.update_one(
                {"person_id": person.person_id},
                {"$set": {
                    "label": person.label,
                    "relationship": person.relationship,
                    "risk_decay_factor": person.risk_decay_factor,
                    "last_seen": person.last_seen,
                    "notes": person.notes,
                    "updated_at": datetime.now()
                }}
            )
            # Update consent history
            self.consent_history.update_one(
                {"person_id": person.person_id},
                {"$set": {
                    "times_appeared": person.consent_history.times_appeared,
                    "times_approved": person.consent_history.times_approved,
                    "times_protected": person.consent_history.times_protected,
                    "contexts": person.consent_history.contexts,
                    "last_consent_decision": person.consent_history.last_consent_decision,
                    "consent_confidence": person.consent_history.consent_confidence
                }}
            )
            return True
        
        except Exception as e:
            print(f"Failed to update person: {e}")
            return False
    
    def delete_person(self, person_id: str) -> bool:
        """
        Delete person from database

        Args:
            person_id: Person's unique ID
        
        Returns:
            True if successful
        """
        try:
            # Delete person
            self.persons.delete_one({"person_id": person_id})
            # Delete embeddings
            self.embeddings.delete_many({'person_id': person_id})
            # Delete consent history
            self.consent_history.delete_one({"person_id": person_id})
            print(f"Deleted person: {person_id}")
            return True
        
        except Exception as e:
            print(f"Failed to delete person: {e}")
            return False
    
    def get_all_embeddings(self) -> List[Tuple[str, List]]:
        """
        Get all embeddings for face matching

        Returns:
            List of (person_id, embedding) tuples
        """
        try:
            embedding_docs = self.embeddings.find()
            embeddings = []
            for doc in embedding_docs:
                embeddings.append((
                    doc["person_id"],
                    self.deserialize_embedding(doc["embedding"])
                ))
            return embeddings
        
        except Exception as e:
            print(f"Failed to get embeddings: {e}")
            return []
    
    def search_persons(self, query: str):
        """
        Search persons by label

        Args:
            query: Search query 

        Returns:
            List of PersonEntry objects
        """
        try:
            # Case-insensitive search
            person_docs = self.persons.find({
                "label": {"$regex": query, "$options": "i"}
            })
            persons = []
            for doc in person_docs:
                person = self.get_person(doc[person["id"]])
                if person:
                    persons.append(person)
            return persons
        
        except Exception as e:
            print(f"Failed to search persons: {e}")
            return []

    def get_statistics(self):
        """
        Get database statistics

        Returns:
            Dictionary with stats
        """
        try:
            person_count = self.persons.count_documents({})
            embedding_count = self.embeddings.count_documents({})

            return {
                "total_persons": person_count,
                "total_embeddings": embedding_count,
                "encryption_enabled": self.encryption_enabled,
                "database": self.database_name
            }
        
        except Exception as e:
            print(f"Failed to get statistics: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            print("MongoDB connection closed")



            


                 
        

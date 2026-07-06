from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone

db = SQLAlchemy()

class SuspiciousAccount(db.Model):
    __tablename__ = 'suspicious_accounts'

    id               = db.Column(db.Integer, primary_key=True)
    platform         = db.Column(db.String(50),  nullable=False)
    username         = db.Column(db.String(100), nullable=False)
    followers_count  = db.Column(db.Integer, default=0)
    verified         = db.Column(db.Integer, default=0)
    retweet_count    = db.Column(db.Integer, default=0)
    mention_count    = db.Column(db.Integer, default=0)
    bio_length       = db.Column(db.Integer, default=0)
    has_location     = db.Column(db.Integer, default=0)
    prediction       = db.Column(db.String(20), nullable=False)
    confidence       = db.Column(db.Float,   default=0.0)
    threat_level     = db.Column(db.String(20), default='Low')
    # FIX: Use timezone-aware UTC timestamp (datetime.utcnow is deprecated in Python 3.12+)
    created_at       = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        return {
            'id':              self.id,
            'platform':        self.platform,
            'username':        self.username,
            'followers_count': self.followers_count,
            'verified':        self.verified,
            'retweet_count':   self.retweet_count,
            'mention_count':   self.mention_count,
            'bio_length':      self.bio_length,
            'has_location':    self.has_location,
            'prediction':      self.prediction,
            'confidence':      self.confidence,
            'threat_level':    self.threat_level,
            'created_at':      self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }

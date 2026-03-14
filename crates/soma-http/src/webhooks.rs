//! Webhook dispatcher — delivers graph events to registered HTTP endpoints.

use std::sync::Arc;

use tokio::sync::{RwLock, broadcast};
use tracing::{debug, warn};

use crate::{GraphEvent, WebhookRegistration};

/// Spawn a background task that listens for graph events and delivers them
/// to all matching webhook registrations.
pub fn spawn_dispatcher(
    mut rx: broadcast::Receiver<GraphEvent>,
    webhooks: Arc<RwLock<Vec<WebhookRegistration>>>,
) {
    tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap_or_default();

        loop {
            match rx.recv().await {
                Ok(event) => {
                    let hooks = webhooks.read().await;
                    for hook in hooks.iter() {
                        if hook.events.contains(&"*".to_string())
                            || hook.events.contains(&event.kind)
                        {
                            let client = client.clone();
                            let url = hook.url.clone();
                            let event = event.clone();
                            let secret = hook.secret.clone();
                            tokio::spawn(async move {
                                deliver(&client, &url, &event, secret.as_deref()).await;
                            });
                        }
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    warn!("[webhooks] Dropped {} events (slow consumer)", n);
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    });
}

async fn deliver(
    client: &reqwest::Client,
    url: &str,
    event: &GraphEvent,
    secret: Option<&str>,
) {
    let mut req = client.post(url).json(event);
    if let Some(s) = secret {
        req = req.header("X-Soma-Secret", s);
    }
    match req.send().await {
        Ok(resp) => {
            debug!("[webhooks] Delivered to {} → {}", url, resp.status());
        }
        Err(e) => {
            warn!("[webhooks] Failed to deliver to {}: {}", url, e);
        }
    }
}

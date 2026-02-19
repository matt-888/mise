use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use eyre::{Report, Result, bail, ensure};
use regex::Regex;
use reqwest::StatusCode;
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::{ClientBuilder, IntoUrl, Method, Response};
use std::sync::LazyLock as Lazy;
use tokio::sync::OnceCell;
use tokio_retry::Retry;
use tokio_retry::strategy::{ExponentialBackoff, jitter};
use url::Url;

use crate::cli::version;
use crate::config::Settings;
use crate::file::display_path;
use crate::netrc;
use crate::ui::progress_report::SingleReport;
use crate::ui::time::format_duration;
use crate::{env, file};

#[cfg(not(test))]
pub static HTTP_VERSION_CHECK: Lazy<Client> =
    Lazy::new(|| Client::new(Duration::from_secs(3), ClientKind::VersionCheck).unwrap());

pub static HTTP: Lazy<Client> =
    Lazy::new(|| Client::new(Settings::get().http_timeout(), ClientKind::Http).unwrap());

pub static HTTP_FETCH: Lazy<Client> = Lazy::new(|| {
    Client::new(
        Settings::get().fetch_remote_versions_timeout(),
        ClientKind::Fetch,
    )
    .unwrap()
});

/// In-memory cache for HTTP text responses, useful for requests that are repeated
/// during a single operation (e.g., fetching SHASUMS256.txt for multiple platforms).
/// Each URL gets its own OnceCell to ensure concurrent requests for the same URL
/// wait for the first fetch to complete rather than all fetching simultaneously.
type CachedResult = Arc<OnceCell<Result<String, String>>>;
static HTTP_CACHE: Lazy<Mutex<HashMap<String, CachedResult>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Debug)]
pub struct Client {
    reqwest: reqwest::Client,
    timeout: Duration,
    kind: ClientKind,
}

#[derive(Debug, Clone, Copy)]
enum ClientKind {
    Http,
    Fetch,
    #[allow(dead_code)]
    VersionCheck,
}

impl Client {
    fn new(timeout: Duration, kind: ClientKind) -> Result<Self> {
        Ok(Self {
            reqwest: Self::_new()
                .read_timeout(timeout)
                .connect_timeout(timeout)
                .build()?,
            timeout,
            kind,
        })
    }

    fn _new() -> ClientBuilder {
        let v = &*version::VERSION;
        let shell = env::MISE_SHELL.map(|s| s.to_string()).unwrap_or_default();
        ClientBuilder::new()
            .user_agent(format!("mise/{v} {shell}").trim())
            .gzip(true)
            .zstd(true)
    }

    pub async fn get_bytes<U: IntoUrl>(&self, url: U) -> Result<impl AsRef<[u8]>> {
        let url = url.into_url().unwrap();
        let resp = self.get_async(url.clone()).await?;
        Ok(resp.bytes().await?)
    }

    pub async fn get_async<U: IntoUrl>(&self, url: U) -> Result<Response> {
        let url = url.into_url().unwrap();
        let headers = github_headers(&url);
        self.get_async_with_headers(url, &headers).await
    }

    async fn get_async_with_headers<U: IntoUrl>(
        &self,
        url: U,
        headers: &HeaderMap,
    ) -> Result<Response> {
        ensure!(!Settings::get().offline(), "offline mode is enabled");
        let url = url.into_url().unwrap();
        let resp = self
            .send_with_https_fallback(Method::GET, url, headers, "GET")
            .await?;
        resp.error_for_status_ref()?;
        Ok(resp)
    }

    pub async fn head<U: IntoUrl>(&self, url: U) -> Result<Response> {
        let url = url.into_url().unwrap();
        let headers = github_headers(&url);
        self.head_async_with_headers(url, &headers).await
    }

    pub async fn head_async_with_headers<U: IntoUrl>(
        &self,
        url: U,
        headers: &HeaderMap,
    ) -> Result<Response> {
        ensure!(!Settings::get().offline(), "offline mode is enabled");
        let url = url.into_url().unwrap();
        let resp = self
            .send_with_https_fallback(Method::HEAD, url, headers, "HEAD")
            .await?;
        resp.error_for_status_ref()?;
        Ok(resp)
    }

    pub async fn get_text<U: IntoUrl>(&self, url: U) -> Result<String> {
        self.get_text_with_headers(url, &HeaderMap::new()).await
    }

    pub async fn get_text_with_headers<U: IntoUrl>(
        &self,
        url: U,
        extra_headers: &HeaderMap,
    ) -> Result<String> {
        let mut url = url.into_url().unwrap();
        // Merge GitHub headers with any extra headers provided
        let mut headers = github_headers(&url);
        headers.extend(extra_headers.clone());
        let resp = self.get_async_with_headers(url.clone(), &headers).await?;
        let text = resp.text().await?;
        if text.starts_with("<!DOCTYPE html>") {
            if url.scheme() == "http" {
                // try with https since http may be blocked
                url.set_scheme("https").unwrap();
                return Box::pin(self.get_text_with_headers(url, extra_headers)).await;
            }
            bail!("Got HTML instead of text from {}", url);
        }
        Ok(text)
    }

    /// Like get_text but caches results in memory for the duration of the process.
    /// Useful when the same URL will be requested multiple times (e.g., SHASUMS256.txt
    /// when locking multiple platforms). Concurrent requests for the same URL will
    /// wait for the first fetch to complete.
    pub async fn get_text_cached<U: IntoUrl>(&self, url: U) -> Result<String> {
        let url = url.into_url().unwrap();
        let key = url.to_string();

        // Get or create the OnceCell for this URL
        let cell = {
            let mut cache = HTTP_CACHE.lock().unwrap();
            cache.entry(key).or_default().clone()
        };

        // Initialize the cell if needed - concurrent callers will wait
        let result = cell
            .get_or_init(|| {
                let url = url.clone();
                async move {
                    match self.get_text(url).await {
                        Ok(text) => Ok(text),
                        Err(err) => Err(err.to_string()),
                    }
                }
            })
            .await;

        match result {
            Ok(text) => Ok(text.clone()),
            Err(err) => bail!("{}", err),
        }
    }

    pub async fn get_html<U: IntoUrl>(&self, url: U) -> Result<String> {
        let url = url.into_url().unwrap();
        let resp = self.get_async(url.clone()).await?;
        let html = resp.text().await?;
        if !html.starts_with("<!DOCTYPE html>") {
            bail!("Got non-HTML text from {}", url);
        }
        Ok(html)
    }

    pub async fn json_headers<T, U: IntoUrl>(&self, url: U) -> Result<(T, HeaderMap)>
    where
        T: serde::de::DeserializeOwned,
    {
        let url = url.into_url().unwrap();
        let resp = self.get_async(url).await?;
        let headers = resp.headers().clone();
        let json = resp.json().await?;
        Ok((json, headers))
    }

    pub async fn json_headers_with_headers<T, U: IntoUrl>(
        &self,
        url: U,
        headers: &HeaderMap,
    ) -> Result<(T, HeaderMap)>
    where
        T: serde::de::DeserializeOwned,
    {
        let url = url.into_url().unwrap();
        let resp = self.get_async_with_headers(url, headers).await?;
        let headers = resp.headers().clone();
        let json = resp.json().await?;
        Ok((json, headers))
    }

    pub async fn json<T, U: IntoUrl>(&self, url: U) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        self.json_headers(url).await.map(|(json, _)| json)
    }

    /// Like json but caches raw JSON text in memory for the duration of the process.
    /// Useful when the same URL will be requested multiple times (e.g., zig index.json
    /// when locking multiple platforms). Concurrent requests for the same URL will
    /// wait for the first fetch to complete.
    pub async fn json_cached<T, U: IntoUrl>(&self, url: U) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let text = self.get_text_cached(url).await?;
        Ok(serde_json::from_str(&text)?)
    }

    pub async fn json_with_headers<T, U: IntoUrl>(&self, url: U, headers: &HeaderMap) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        self.json_headers_with_headers(url, headers)
            .await
            .map(|(json, _)| json)
    }

    /// POST JSON data to a URL. Returns Ok(true) on success, Ok(false) on non-success status.
    /// Errors only on network/connection failures.
    #[allow(dead_code)]
    pub async fn post_json<U: IntoUrl, T: serde::Serialize>(
        &self,
        url: U,
        body: &T,
    ) -> Result<bool> {
        self.post_json_with_headers(url, body, &HeaderMap::new())
            .await
    }

    /// POST JSON data to a URL with custom headers.
    pub async fn post_json_with_headers<U: IntoUrl, T: serde::Serialize>(
        &self,
        url: U,
        body: &T,
        headers: &HeaderMap,
    ) -> Result<bool> {
        ensure!(!Settings::get().offline(), "offline mode is enabled");
        let url = url.into_url()?;
        debug!("POST {}", &url);
        let resp = self
            .reqwest
            .post(url)
            .header("Content-Type", "application/json")
            .headers(headers.clone())
            .json(body)
            .send()
            .await?;
        Ok(resp.status().is_success())
    }

    pub async fn download_file<U: IntoUrl>(
        &self,
        url: U,
        path: &Path,
        pr: Option<&dyn SingleReport>,
    ) -> Result<()> {
        let url = url.into_url()?;
        let headers = github_headers(&url);
        self.download_file_with_headers(url, path, &headers, pr)
            .await
    }

    pub async fn download_file_with_headers<U: IntoUrl>(
        &self,
        url: U,
        path: &Path,
        headers: &HeaderMap,
        pr: Option<&dyn SingleReport>,
    ) -> Result<()> {
        let url = url.into_url()?;
        debug!("GET Downloading {} to {}", &url, display_path(path));

        let parent = path.parent().unwrap();
        file::create_dir_all(parent)?;
        let part_path = part_path_for(path);

        let max_retries = Settings::get().http_retries.max(3) as u32;
        let mut attempt = 0u32;
        let mut last_error: Option<Report> = None;

        loop {
            if attempt > max_retries {
                break;
            }

            if attempt > 0 {
                let delay = download_backoff_delay(attempt);
                debug!(
                    "Retrying download (attempt {}/{}) after {}ms",
                    attempt,
                    max_retries,
                    delay.as_millis()
                );
                tokio::time::sleep(delay).await;
            }

            let bytes_downloaded = match std::fs::metadata(&part_path) {
                Ok(m) => m.len(),
                Err(_) => 0,
            };

            // Build request with Range header if resuming
            let mut req_headers = headers.clone();
            if bytes_downloaded > 0 {
                req_headers.insert(
                    reqwest::header::RANGE,
                    HeaderValue::from_str(&format!("bytes={bytes_downloaded}-")).unwrap(),
                );
                debug!("Resuming download from byte {}", bytes_downloaded);
            }

            let resp = match self
                .send_with_https_fallback(Method::GET, url.clone(), &req_headers, "GET")
                .await
            {
                Ok(resp) => resp,
                Err(e) => {
                    // Check for 416 Range Not Satisfiable — part file is stale
                    if error_code(&e) == Some(416) {
                        debug!("Got 416 Range Not Satisfiable, deleting part file and restarting");
                        let _ = std::fs::remove_file(&part_path);
                        // Don't count this as a retry attempt
                        continue;
                    }
                    last_error = Some(e);
                    attempt += 1;
                    continue;
                }
            };

            let status = resp.status();
            let (mut file, start_pos) = if status == StatusCode::PARTIAL_CONTENT {
                // Server supports Range, append to existing part file
                let total = parse_content_range_total(resp.headers());
                if let Some(pr) = pr {
                    if let Some(total) = total {
                        pr.set_length(total);
                    }
                    pr.set_position(bytes_downloaded);
                }
                let file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&part_path)?;
                (file, bytes_downloaded)
            } else {
                // 200 OK — server sent full file, start from scratch
                if let Some(pr) = pr {
                    if let Some(length) = resp.content_length() {
                        pr.set_length(length);
                    }
                    pr.set_position(0);
                }
                let file = std::fs::File::create(&part_path)?;
                (file, 0u64)
            };

            // Stream the response body
            let stream_result = self
                .stream_response_to_file(resp, &mut file, pr, start_pos)
                .await;

            drop(file);

            match stream_result {
                Ok(()) => {
                    // Success — atomic move to final destination
                    std::fs::rename(&part_path, path)?;
                    return Ok(());
                }
                Err(e) => {
                    debug!("Download stream error: {e}");
                    last_error = Some(e);
                    attempt += 1;
                    continue;
                }
            }
        }

        // All retries exhausted — clean up part file
        let _ = std::fs::remove_file(&part_path);
        Err(last_error
            .unwrap_or_else(|| eyre::eyre!("download failed after {max_retries} retries")))
    }

    /// Stream response body chunks to a file, updating progress bar.
    async fn stream_response_to_file(
        &self,
        mut resp: Response,
        file: &mut std::fs::File,
        pr: Option<&dyn SingleReport>,
        start_pos: u64,
    ) -> Result<()> {
        let mut written = start_pos;
        while let Some(chunk) = resp.chunk().await? {
            file.write_all(&chunk)?;
            written += chunk.len() as u64;
            if let Some(pr) = pr {
                pr.set_position(written);
            }
        }
        Ok(())
    }

    async fn send_with_https_fallback(
        &self,
        method: Method,
        url: Url,
        headers: &HeaderMap,
        verb_label: &str,
    ) -> Result<Response> {
        Retry::spawn(
            default_backoff_strategy(Settings::get().http_retries),
            || {
                let method = method.clone();
                let url = url.clone();
                let headers = headers.clone();
                async move {
                    match self
                        .send_once(method.clone(), url.clone(), &headers, verb_label)
                        .await
                    {
                        Ok(resp) => Ok(resp),
                        Err(_err) if url.scheme() == "http" => {
                            let mut url = url;
                            url.set_scheme("https").unwrap();
                            self.send_once(method, url, &headers, verb_label).await
                        }
                        Err(err) => Err(err),
                    }
                }
            },
        )
        .await
    }

    async fn send_once(
        &self,
        method: Method,
        mut url: Url,
        headers: &HeaderMap,
        verb_label: &str,
    ) -> Result<Response> {
        apply_url_replacements(&mut url);
        debug!("{} {}", verb_label, &url);

        // Apply netrc credentials after URL replacement
        let mut final_headers = headers.clone();
        final_headers.extend(netrc_headers(&url));

        let mut req = self.reqwest.request(method, url.clone());
        req = req.headers(final_headers);
        let resp = match req.send().await {
            Ok(resp) => resp,
            Err(err) => {
                if err.is_timeout() {
                    let (setting, env_var) = match self.kind {
                        ClientKind::Http => ("http_timeout", "MISE_HTTP_TIMEOUT"),
                        ClientKind::Fetch => (
                            "fetch_remote_versions_timeout",
                            "MISE_FETCH_REMOTE_VERSIONS_TIMEOUT",
                        ),
                        ClientKind::VersionCheck => ("version_check_timeout", ""),
                    };
                    let hint = if env_var.is_empty() {
                        format!(
                            "HTTP timed out after {} for {}.",
                            format_duration(self.timeout),
                            url
                        )
                    } else {
                        format!(
                            "HTTP timed out after {} for {} (change with `{}` or env `{}`).",
                            format_duration(self.timeout),
                            url,
                            setting,
                            env_var
                        )
                    };
                    bail!(hint);
                }
                return Err(err.into());
            }
        };
        if *env::MISE_LOG_HTTP {
            eprintln!("{} {url} {}", verb_label, resp.status());
        }
        debug!("{} {url} {}", verb_label, resp.status());
        display_github_rate_limit(&resp);
        resp.error_for_status_ref()?;
        Ok(resp)
    }
}

pub fn error_code(e: &Report) -> Option<u16> {
    if e.to_string().contains("404") {
        // TODO: not this when I can figure out how to use eyre properly
        return Some(404);
    }
    if let Some(err) = e.downcast_ref::<reqwest::Error>() {
        err.status().map(|s| s.as_u16())
    } else {
        None
    }
}

fn github_headers(url: &Url) -> HeaderMap {
    let mut headers = HeaderMap::new();
    if url.host_str() == Some("api.github.com")
        && let Some(token) = &*env::GITHUB_TOKEN
    {
        headers.insert(
            reqwest::header::AUTHORIZATION,
            HeaderValue::from_str(format!("Bearer {token}").as_str()).unwrap(),
        );
        headers.insert(
            "x-github-api-version",
            HeaderValue::from_static("2022-11-28"),
        );
    }
    headers
}

/// Get HTTP Basic authentication headers from netrc file for the given URL
fn netrc_headers(url: &Url) -> HeaderMap {
    let mut headers = HeaderMap::new();
    if let Some(host) = url.host_str()
        && let Some((login, password)) = netrc::get_credentials(host)
    {
        let credentials = BASE64_STANDARD.encode(format!("{login}:{password}"));
        if let Ok(value) = HeaderValue::from_str(&format!("Basic {credentials}")) {
            headers.insert(reqwest::header::AUTHORIZATION, value);
        }
    }
    headers
}

/// Apply URL replacements based on settings configuration
/// Supports both simple string replacement and regex patterns (prefixed with "regex:")
pub fn apply_url_replacements(url: &mut Url) {
    let settings = Settings::get();
    if let Some(replacements) = &settings.url_replacements {
        let url_string = url.to_string();

        for (pattern, replacement) in replacements {
            if let Some(pattern_without_prefix) = pattern.strip_prefix("regex:") {
                // Regex replacement
                if let Ok(regex) = Regex::new(pattern_without_prefix) {
                    let new_url_string = regex.replace(&url_string, replacement.as_str());
                    // Only proceed if the URL actually changed
                    if new_url_string != url_string
                        && let Ok(new_url) = new_url_string.parse()
                    {
                        *url = new_url;
                        trace!(
                            "Replaced URL using regex '{}': {} -> {}",
                            pattern_without_prefix,
                            url_string,
                            url.as_str()
                        );
                        return; // Apply only the first matching replacement
                    }
                } else {
                    warn!(
                        "Invalid regex pattern in URL replacement: {}",
                        pattern_without_prefix
                    );
                }
            } else {
                // Simple string replacement
                if url_string.contains(pattern) {
                    let new_url_string = url_string.replace(pattern, replacement);
                    // Only proceed if the URL actually changed
                    if new_url_string != url_string
                        && let Ok(new_url) = new_url_string.parse()
                    {
                        *url = new_url;
                        trace!(
                            "Replaced URL using string replacement '{}': {} -> {}",
                            pattern,
                            url_string,
                            url.as_str()
                        );
                        return; // Apply only the first matching replacement
                    }
                }
            }
        }
    }
}

fn display_github_rate_limit(resp: &Response) {
    let status = resp.status().as_u16();
    if status == 403 || status == 429 {
        let remaining = resp
            .headers()
            .get("x-ratelimit-remaining")
            .and_then(|r| r.to_str().ok());
        if remaining.is_some_and(|r| r == "0") {
            if let Some(reset_time) = resp
                .headers()
                .get("x-ratelimit-reset")
                .and_then(|h| h.to_str().ok())
                .and_then(|s| s.parse::<i64>().ok())
                .and_then(|ts| chrono::DateTime::from_timestamp(ts, 0))
            {
                warn!(
                    "GitHub rate limit exceeded. Resets at {}",
                    reset_time.with_timezone(&chrono::Local)
                );
            }
            return;
        }
        // retry-after header is processed only if x-ratelimit-remaining is not 0 or is missing
        if let Some(retry_after) = resp
            .headers()
            .get("retry-after")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
        {
            warn!(
                "GitHub rate limit exceeded. Retry after {} seconds",
                retry_after
            );
        }
    }
}

fn default_backoff_strategy(retries: i64) -> impl Iterator<Item = std::time::Duration> {
    ExponentialBackoff::from_millis(10)
        .map(jitter)
        .take(retries.max(0) as usize)
}

/// Compute the `.part` file path for a given download destination.
fn part_path_for(path: &Path) -> PathBuf {
    let mut part = path.as_os_str().to_owned();
    part.push(".part");
    PathBuf::from(part)
}

/// Parse the total file size from a `Content-Range` header.
/// Expected format: `bytes <start>-<end>/<total>` (e.g. `bytes 0-999/5000`).
fn parse_content_range_total(headers: &HeaderMap) -> Option<u64> {
    let value = headers.get(reqwest::header::CONTENT_RANGE)?.to_str().ok()?;
    let total_str = value.rsplit_once('/')?.1;
    if total_str == "*" {
        return None;
    }
    total_str.parse().ok()
}

/// Exponential backoff delay for download retries, capped at 30 seconds.
fn download_backoff_delay(attempt: u32) -> Duration {
    let base_ms = 500u64.saturating_mul(1u64 << attempt.min(6));
    Duration::from_millis(base_ms.min(30_000))
}

#[cfg(test)]
mod tests {
    use super::*;
    use confique::Partial;
    use indexmap::IndexMap;
    use url::Url;

    // Mutex to ensure tests don't interfere with each other when modifying global settings
    static TEST_SETTINGS_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    // Helper to create test settings with specific URL replacements
    fn with_test_settings<F, R>(replacements: IndexMap<String, String>, test_fn: F) -> R
    where
        F: FnOnce() -> R,
    {
        // Lock to prevent parallel tests from interfering with global settings
        let _guard = TEST_SETTINGS_LOCK.lock().unwrap();

        // Create settings with custom URL replacements
        let mut settings = crate::config::settings::SettingsPartial::empty();
        settings.url_replacements = Some(replacements);

        // Set settings for this test
        crate::config::Settings::reset(Some(settings));

        // Run test
        let result = test_fn();

        // Clean up after test
        crate::config::Settings::reset(None);

        result
    }

    #[test]
    fn test_simple_string_replacement() {
        let mut replacements = IndexMap::new();
        replacements.insert("github.com".to_string(), "my-proxy.com".to_string());

        with_test_settings(replacements, || {
            let mut url = Url::parse("https://github.com/owner/repo").unwrap();
            apply_url_replacements(&mut url);
            assert_eq!(url.as_str(), "https://my-proxy.com/owner/repo");
        });
    }

    #[test]
    fn test_full_url_string_replacement() {
        let mut replacements = IndexMap::new();
        replacements.insert(
            "https://github.com".to_string(),
            "https://my-proxy.com/artifactory/github-remote".to_string(),
        );

        with_test_settings(replacements, || {
            let mut url = Url::parse("https://github.com/owner/repo").unwrap();
            apply_url_replacements(&mut url);
            assert_eq!(
                url.as_str(),
                "https://my-proxy.com/artifactory/github-remote/owner/repo"
            );
        });
    }

    #[test]
    fn test_protocol_specific_replacement() {
        let mut replacements = IndexMap::new();
        replacements.insert(
            "https://github.com".to_string(),
            "https://secure-proxy.com".to_string(),
        );

        with_test_settings(replacements.clone(), || {
            // HTTPS gets replaced
            let mut url1 = Url::parse("https://github.com/owner/repo").unwrap();
            apply_url_replacements(&mut url1);
            assert_eq!(url1.as_str(), "https://secure-proxy.com/owner/repo");
        });

        with_test_settings(replacements, || {
            // HTTP does not get replaced (no match)
            let mut url2 = Url::parse("http://github.com/owner/repo").unwrap();
            apply_url_replacements(&mut url2);
            assert_eq!(url2.as_str(), "http://github.com/owner/repo");
        });
    }

    #[test]
    fn test_regex_replacement() {
        let mut replacements = IndexMap::new();
        replacements.insert(
            r"regex:https://github\.com".to_string(),
            "https://my-proxy.com".to_string(),
        );

        with_test_settings(replacements, || {
            let mut url = Url::parse("https://github.com/owner/repo").unwrap();
            apply_url_replacements(&mut url);
            assert_eq!(url.as_str(), "https://my-proxy.com/owner/repo");
        });
    }

    #[test]
    fn test_regex_with_capture_groups() {
        let mut replacements = IndexMap::new();
        replacements.insert(
            r"regex:https://github\.com/([^/]+)/([^/]+)".to_string(),
            "https://my-proxy.com/mirror/$1/$2".to_string(),
        );

        with_test_settings(replacements, || {
            let mut url = Url::parse("https://github.com/owner/repo/releases").unwrap();
            apply_url_replacements(&mut url);
            assert_eq!(
                url.as_str(),
                "https://my-proxy.com/mirror/owner/repo/releases"
            );
        });
    }

    #[test]
    fn test_regex_invalid_replacement_url() {
        let mut replacements = IndexMap::new();
        replacements.insert(
            r"regex:https://github\.com/([^/]+)".to_string(),
            "not-a-valid-url".to_string(),
        );

        with_test_settings(replacements, || {
            // Invalid result URL should be ignored, original URL unchanged
            let mut url = Url::parse("https://github.com/owner/repo").unwrap();
            let original = url.clone();
            apply_url_replacements(&mut url);
            assert_eq!(url.as_str(), original.as_str());
        });
    }

    #[test]
    fn test_multiple_replacements_first_match_wins() {
        let mut replacements = IndexMap::new();
        replacements.insert("github.com".to_string(), "first-proxy.com".to_string());
        replacements.insert("github".to_string(), "second-proxy.com".to_string());

        with_test_settings(replacements, || {
            let mut url = Url::parse("https://github.com/owner/repo").unwrap();
            apply_url_replacements(&mut url);
            // First replacement should win
            assert_eq!(url.as_str(), "https://first-proxy.com/owner/repo");
        });
    }

    #[test]
    fn test_no_replacements_configured() {
        let replacements = IndexMap::new(); // Empty

        with_test_settings(replacements, || {
            let mut url = Url::parse("https://github.com/owner/repo").unwrap();
            let original = url.clone();
            apply_url_replacements(&mut url);
            assert_eq!(url.as_str(), original.as_str());
        });
    }

    #[test]
    fn test_regex_complex_patterns() {
        let mut replacements = IndexMap::new();
        // Convert GitHub releases to JFrog Artifactory
        replacements.insert(
            r"regex:https://github\.com/([^/]+)/([^/]+)/releases/download/([^/]+)/(.+)".to_string(),
            "https://artifactory.company.com/artifactory/github-releases/$1/$2/$3/$4".to_string(),
        );

        with_test_settings(replacements, || {
            let mut url =
                Url::parse("https://github.com/owner/repo/releases/download/v1.0.0/file.tar.gz")
                    .unwrap();
            apply_url_replacements(&mut url);
            assert_eq!(
                url.as_str(),
                "https://artifactory.company.com/artifactory/github-releases/owner/repo/v1.0.0/file.tar.gz"
            );
        });
    }

    #[test]
    fn test_no_settings_configured() {
        // Test the real apply_url_replacements function with no settings override
        let _guard = TEST_SETTINGS_LOCK.lock().unwrap();
        crate::config::Settings::reset(None);

        let mut url = Url::parse("https://github.com/owner/repo").unwrap();
        let original = url.clone();

        // This should not crash and should leave URL unchanged
        apply_url_replacements(&mut url);
        assert_eq!(url.as_str(), original.as_str());
    }

    #[test]
    fn test_replacement_affects_full_url_not_just_hostname() {
        // Test that replacement works on the full URL string, not just hostname
        let mut replacements = IndexMap::new();
        replacements.insert(
            "github.com/owner".to_string(),
            "proxy.com/mirror".to_string(),
        );

        with_test_settings(replacements, || {
            let mut url = Url::parse("https://github.com/owner/repo").unwrap();
            apply_url_replacements(&mut url);
            // This demonstrates that replacement happens on full URL, not just hostname
            assert_eq!(url.as_str(), "https://proxy.com/mirror/repo");
        });
    }

    #[test]
    fn test_path_replacement_example() {
        // Test replacing part of the path, proving it's not hostname-only
        let mut replacements = IndexMap::new();
        replacements.insert("/releases/download/".to_string(), "/artifacts/".to_string());

        with_test_settings(replacements, || {
            let mut url =
                Url::parse("https://github.com/owner/repo/releases/download/v1.0.0/file.tar.gz")
                    .unwrap();
            apply_url_replacements(&mut url);
            // Path component was replaced, proving it's full URL replacement
            assert_eq!(
                url.as_str(),
                "https://github.com/owner/repo/artifacts/v1.0.0/file.tar.gz"
            );
        });
    }

    #[test]
    fn test_documentation_examples() {
        // Test the examples from the documentation to ensure they work correctly

        // Example 1: Simple hostname replacement
        let mut replacements = IndexMap::new();
        replacements.insert("github.com".to_string(), "myregistry.net".to_string());

        with_test_settings(replacements, || {
            let mut url = Url::parse("https://github.com/user/repo").unwrap();
            apply_url_replacements(&mut url);
            assert_eq!(url.as_str(), "https://myregistry.net/user/repo");
        });

        // Example 2: Protocol + hostname replacement
        let mut replacements2 = IndexMap::new();
        replacements2.insert(
            "https://github.com".to_string(),
            "https://proxy.corp.com/github-mirror".to_string(),
        );

        with_test_settings(replacements2, || {
            let mut url = Url::parse("https://github.com/user/repo").unwrap();
            apply_url_replacements(&mut url);
            assert_eq!(
                url.as_str(),
                "https://proxy.corp.com/github-mirror/user/repo"
            );
        });

        // Example 3: Domain + path replacement
        let mut replacements3 = IndexMap::new();
        replacements3.insert(
            "github.com/releases/download/".to_string(),
            "cdn.example.com/artifacts/".to_string(),
        );

        with_test_settings(replacements3, || {
            let mut url =
                Url::parse("https://github.com/releases/download/v1.0.0/file.tar.gz").unwrap();
            apply_url_replacements(&mut url);
            assert_eq!(
                url.as_str(),
                "https://cdn.example.com/artifacts/v1.0.0/file.tar.gz"
            );
        });
    }

    #[test]
    fn test_part_path_for() {
        let path = Path::new("/tmp/downloads/python-3.12.tar.gz");
        assert_eq!(
            part_path_for(path),
            PathBuf::from("/tmp/downloads/python-3.12.tar.gz.part")
        );
    }

    #[test]
    fn test_parse_content_range_total() {
        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_RANGE,
            HeaderValue::from_static("bytes 1000-1999/5000"),
        );
        assert_eq!(parse_content_range_total(&headers), Some(5000));
    }

    #[test]
    fn test_parse_content_range_total_unknown() {
        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_RANGE,
            HeaderValue::from_static("bytes 0-999/*"),
        );
        assert_eq!(parse_content_range_total(&headers), None);
    }

    #[test]
    fn test_parse_content_range_total_missing() {
        let headers = HeaderMap::new();
        assert_eq!(parse_content_range_total(&headers), None);
    }

    #[test]
    fn test_download_backoff_delay() {
        // First retry: 500 * 2^1 = 1000ms
        assert_eq!(download_backoff_delay(1), Duration::from_millis(1000));
        // Second retry: 500 * 2^2 = 2000ms
        assert_eq!(download_backoff_delay(2), Duration::from_millis(2000));
        // Third retry: 500 * 2^3 = 4000ms
        assert_eq!(download_backoff_delay(3), Duration::from_millis(4000));
        // High attempt should cap at 30 seconds
        assert_eq!(download_backoff_delay(10), Duration::from_millis(30_000));
    }
}

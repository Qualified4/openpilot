// C ABI over msgq / VisionIPC (msgq_repo) for the Rust daemon (src/ipc.rs). Thin: no logic beyond the calls.
#include <cstring>
#include <string>
#include <vector>

#include "msgq/ipc.h"
#include "msgq/visionipc/visionipc_client.h"

extern "C" {

struct sn_sub { Context *ctx; SubSocket *sock; Poller *poller; Message *last; };
struct sn_pub { Context *ctx; PubSocket *sock; };

struct sn_frame {
  const uint8_t *data; size_t len; size_t width, height, stride, uv_offset;
  uint32_t frame_id; uint64_t timestamp_eof; void *buf;
};

sn_sub *sn_sub_new(const char *service, int conflate) {
  Context *ctx = Context::create();
  SubSocket *sock = SubSocket::create(ctx, service, "127.0.0.1", conflate != 0, true);
  if (sock == nullptr) { delete ctx; return nullptr; }
  Poller *poller = Poller::create({sock});
  return new sn_sub{ctx, sock, poller, nullptr};
}

// 1 when a message arrived within timeout_ms (poll), else 0.
int sn_sub_poll(sn_sub *s, int timeout_ms) { return s->poller->poll(timeout_ms).empty() ? 0 : 1; }

// The next message without waiting (data valid until the next call on this socket), or nullptr.
const uint8_t *sn_sub_recv(sn_sub *s, size_t *len) {
  delete s->last;
  s->last = s->sock->receive(true);
  if (s->last == nullptr) return nullptr;
  *len = s->last->getSize();
  return reinterpret_cast<const uint8_t *>(s->last->getData());
}

void sn_sub_free(sn_sub *s) { delete s->last; delete s->poller; delete s->sock; delete s->ctx; delete s; }

sn_pub *sn_pub_new(const char *service) {
  Context *ctx = Context::create();
  PubSocket *sock = PubSocket::create(ctx, service, true);
  if (sock == nullptr) { delete ctx; return nullptr; }
  return new sn_pub{ctx, sock};
}

int sn_pub_send(sn_pub *p, const uint8_t *data, size_t len) {
  return p->sock->send(reinterpret_cast<char *>(const_cast<uint8_t *>(data)), len);
}

void sn_pub_free(sn_pub *p) { delete p->sock; delete p->ctx; delete p; }

void *sn_vipc_new(const char *name, int stream, int conflate) {
  return new VisionIpcClient(name, static_cast<VisionStreamType>(stream), conflate != 0);
}
int sn_vipc_connect(void *c, int blocking) { return static_cast<VisionIpcClient *>(c)->connect(blocking != 0) ? 1 : 0; }
int sn_vipc_connected(void *c) { return static_cast<VisionIpcClient *>(c)->is_connected() ? 1 : 0; }

// One buffer (frame id / EOF from the packet extra) or 0 on timeout.
int sn_vipc_recv(void *c, int timeout_ms, sn_frame *out) {
  VisionIpcBufExtra extra = {};
  VisionBuf *b = static_cast<VisionIpcClient *>(c)->recv(&extra, timeout_ms);
  if (b == nullptr) return 0;
  *out = sn_frame{static_cast<const uint8_t *>(b->addr), b->len, b->width, b->height, b->stride, b->uv_offset,
                  extra.frame_id, extra.timestamp_eof, b};
  return 1;
}

// The frame id camerad currently stores in a held buffer (buffer_drop_reason's slot check).
uint64_t sn_vipc_buf_frame_id(void *buf) { return static_cast<VisionBuf *>(buf)->get_frame_id(); }

void sn_vipc_free(void *c) { delete static_cast<VisionIpcClient *>(c); }

}

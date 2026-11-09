// LD_PRELOAD shim: bind() to PORT_REMAP_FROM goes to PORT_REMAP_TO instead.
// The AirSim plugin in the AirVLN scene ignores ApiServerPort and always binds 41451;
// this lets a second renderer share the host without touching the first one.
// Build: gcc -shared -fPIC -O2 -o port_remap_shim.so port_remap_shim.c -ldl
#define _GNU_SOURCE
#include <dlfcn.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

int bind(int fd, const struct sockaddr *addr, socklen_t len) {
    static int (*real_bind)(int, const struct sockaddr *, socklen_t) = NULL;
    if (!real_bind) real_bind = dlsym(RTLD_NEXT, "bind");
    const char *f = getenv("PORT_REMAP_FROM"), *t = getenv("PORT_REMAP_TO");
    if (!f || !t || !addr) return real_bind(fd, addr, len);
    unsigned short from = (unsigned short)atoi(f), to = (unsigned short)atoi(t);
    if (addr->sa_family == AF_INET && len >= sizeof(struct sockaddr_in)) {
        struct sockaddr_in a;
        memcpy(&a, addr, sizeof a);
        if (ntohs(a.sin_port) == from) a.sin_port = htons(to);
        return real_bind(fd, (struct sockaddr *)&a, sizeof a);
    }
    if (addr->sa_family == AF_INET6 && len >= sizeof(struct sockaddr_in6)) {
        struct sockaddr_in6 a;
        memcpy(&a, addr, sizeof a);
        if (ntohs(a.sin6_port) == from) a.sin6_port = htons(to);
        return real_bind(fd, (struct sockaddr *)&a, sizeof a);
    }
    return real_bind(fd, addr, len);
}

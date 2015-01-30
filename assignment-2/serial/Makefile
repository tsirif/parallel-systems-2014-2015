OPTI = -O3 -finline-functions -fomit-frame-pointer -DNDEBUG \
-fno-strict-aliasing --param max-inline-insns-single=1800
STD = -std=c99
WARN = -Wmissing-prototypes -Wall #-Winline
CC = gcc
CCFLAGS = $(OPTI) $(WARN) $(STD)

serial-bitonic: bitonic.c
	$(CC) $(CCFLAGS) -o ../$@ $<
